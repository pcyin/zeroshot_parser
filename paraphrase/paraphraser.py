import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict
import subprocess
import ujson
from random import Random
from datetime import datetime

import torch
import torch.nn as nn

from fairseq.models.bart import BARTModel


class Paraphraser(nn.Module):
    def generate(
        self,
        batch_source: List[str],
        beam_size: int,
        **kwargs
    ) -> List[List[Dict]]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> 'Paraphraser':
        if Path(model_name_or_path).name == 'paraphraser-bart-large-speedup-megabatch-5m':
            print('Creating genienlp paraphraser')
            return GenieNlpParaphraser(model_name_or_path)
        elif Path(model_name_or_path).name.startswith('diverse_paraphrasing_inference'):
            print('Creating pseudo decode file paraphraser')
            return DecodeFileParaphraser(model_name_or_path)
        elif Path(model_name_or_path).name.endswith('.jsonl'):
            print('Creating pseudo decode file paraphraser [new version]')
            return DecodeFileParaphraser(model_name_or_path)
        else:
            print('Creating own paraphraser')
            return BartParaphraser(model_name_or_path)


class BartParaphraser(Paraphraser):
    """A paraphraser based on pre-trained BART models in fairseq."""

    def __init__(
        self,
        model_path: str
    ):
        super().__init__()

        model_path = Path(model_path)
        self.model = BARTModel.from_pretrained(
            str(model_path.parent),
            checkpoint_file=str(model_path.name),
        )

        if torch.cuda.is_available():
            print('use gpu.')
            self.model.cuda()

        self.model.eval()

    def generate(
        self,
        batch_source: List[str],
        beam_size: int,
        **kwargs
    ) -> List[List[Dict]]:
        hypotheses_batch = self.model.sample(
            batch_source,
            beam=beam_size,
            **kwargs
        )

        return hypotheses_batch


class GenieNlpParaphraser(Paraphraser):
    """Generate paraphrases using genie_nlp."""

    def __init__(self, model_path: str):
        super().__init__()

        model_path = Path(model_path)
        assert model_path.exists()

        conda_init_script_path = Path(os.environ['CONDA_EXE'])
        assert str(conda_init_script_path).endswith('/bin/conda')
        self.script_path = conda_init_script_path.parent.parent / 'envs' / 'genie' / 'bin' / 'python'
        assert self.script_path.exists()

        self.model_path = model_path

    def generate(
        self,
        batch_source: List[str],
        beam_size: int,
        **kwargs
    ):
        rng = Random()
        rng.seed(datetime.now())

        rand_num = rng.randint(0, 99999)
        tmp_input_file = Path(f'tmp/genie_input_sents.{rand_num}.tsv')
        assert not tmp_input_file.exists()

        tmp_input_file.parent.mkdir(parents=True, exist_ok=True)
        with tmp_input_file.open('w') as f:
            for sent in batch_source:
                f.write(sent + '\n')

        rng = Random()
        rng.seed(datetime.now())
        rand_num = rng.randint(0, 99999)
        tmp_intermediate_file = Path(f'tmp/genie_intermediate.{rand_num}.tsv')

        args = [
            self.script_path,
            self.script_path.with_name('genienlp'),
            'run-paraphrase',
            '--task', 'paraphrase',
            '--model_name_or_path', self.model_path,
            '--input_file', tmp_input_file,
            '--intermediate_file', tmp_intermediate_file,
            '--input_column', 0,
            # '--temperature', kwargs.get('temperature', 0),
            # '--top_p', kwargs.get('top_p', 1.0),
            '--batch_size', 32,
            '--skip_heuristics',
        ]

        if kwargs:
            for key, val in kwargs.items():
                args.append('--' + key)
                if key in {'temperature', 'top_p'}:
                    args.extend(str(val).strip().split(' '))
                else:
                    args.append(val)

        if beam_size > 0:
            args.extend(['--num_beams', beam_size])

        args = [str(x) for x in args]
        print('Args to the paraphraser', ' '.join(args))

        try:
            process = subprocess.run(
                args,
                capture_output=True,
                check=True,
                #stderr=sys.stderr
            )
        except subprocess.CalledProcessError as e:
            print(e.output)
            print(e.stderr)
            raise e

        print(process.stderr.decode("utf-8"))
        output = process.stdout.decode("utf-8")
        # print(output)
        result = ujson.loads(output)
        assert len(result) == len(batch_source)

        print('First paraphrased examples:')
        print(result[:20])

        decode_result = []
        for hyp_list in result:
            hyps = []
            for sent in hyp_list:
                hyp = {
                    'sentence': sent,
                    'score': -0.
                }

                hyps.append(hyp)

            decode_result.append(hyps)

        os.remove(str(tmp_input_file))

        return decode_result


class DecodeFileParaphraser(Paraphraser):
    """Generate paraphrases from a decode file."""

    def __init__(self, model_path: str):
        super().__init__()

        self.decode_file_path = Path(model_path)
        assert self.decode_file_path.exists()

        decode_dict = {}
        for line in self.decode_file_path.open():
            entry = json.loads(line)

            is_old_version = 'input' in entry
            if is_old_version:
                src_sent = entry['input']['inputs_pretokenized'].partition('input: ')[-1].strip().lower()
                tgt_sent = entry['prediction'].strip()
                decode_dict[src_sent] = [tgt_sent]
            else:
                src_sent = entry['text'].strip().lower()
                decode_dict[src_sent] = dict()
                for predictions_entry in entry['pp_list']:
                    lex = predictions_entry['lex']
                    syntax = predictions_entry['syntax']

                    decode_dict[src_sent][(lex, syntax)] = predictions_entry['predictions']

        self.decode_dict = decode_dict
        self.cum_query_count = 0
        self.cum_hit = 0
        self.logged = False

    def generate(
        self,
        batch_source: List[str],
        beam_size: int,
        **kwargs
    ) -> List[List[Dict]]:
        decode_result = []
        lex = kwargs.get('lex', 20)
        syntax = kwargs.get('syntax', 60)

        if not self.logged:
            print(f'Paraphraser args: beam_size={beam_size}, {kwargs}')
            self.logged = True

        for src_sent in batch_source:
            hyps = []
            key = src_sent.strip().lower()
            if key in self.decode_dict:
                tgt_sents = self.decode_dict[key][(lex, syntax)][:beam_size]
                for tgt_sent in tgt_sents:
                    hyp = {
                        'sentence': tgt_sent,
                        'score': -0.
                    }
                    # print(hyp)
                    hyps.append(hyp)
                self.cum_hit += 1

            self.cum_query_count += 1

            decode_result.append(hyps)

        if self.cum_query_count % 50 == 0:
            print(f'Paraphraser: hit ratio '
                  f'{self.cum_hit}/{self.cum_query_count}'
                  f'={self.cum_hit / self.cum_query_count:.2f}')

        return decode_result
