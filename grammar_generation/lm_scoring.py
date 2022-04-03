import json
import math
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GPT2Tokenizer


class PretrainedLM(object):
    def __init__(self, model_name: str, device: torch.device):
        print(f'Loading model {model_name}')
        model_name = str(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
            # self.tokenizer.pad_token = "<|pad|>"
            # self.model.resize_token_embeddings(len(self.tokenizer)) FIXME: why this cause worse LM prob?

        self.model.eval()
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            self.model = self.model.to(device)

        print(f'Worker is using {device}')

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def get_sentence_likelihood(self, sentences: List[str], append_eos: bool = True, return_dict: bool = False):
        sentences = [
            self.tokenizer.bos_token + sent + (self.tokenizer.eos_token if append_eos else '')
            # sent
            for sent
            in sentences
        ]

        batch_encoding = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True
        )

        batch_encoding = {
            k: v.to(self.device)
            for k, v
            in batch_encoding.items()
        }

        input_tensor = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']
        max_input_length = input_tensor.size(-1)

        input_tensor.masked_fill_(input_tensor == self.tokenizer.pad_token_id,  0)

        with torch.no_grad():
            logits = self.model(input_tensor, attention_mask=attention_mask)[0]

            # (batch_size, max_input_length - 1, )
            shifted_lm_logits = logits[..., :-1, :]
            # (batch_size, max_input_length - 1, )
            targets = input_tensor[..., 1:]
            targets_mask = attention_mask[:, 1:]

            # print(input_tensor)

            target_tokens_nll = nn.CrossEntropyLoss(reduction='none')(
                shifted_lm_logits.reshape(-1, shifted_lm_logits.size(-1)),
                targets.flatten()
            ) * targets_mask.flatten()
            target_tokens_nll = target_tokens_nll.view(len(sentences), max_input_length - 1)

            sentence_nll_gpu = target_tokens_nll.sum(dim=-1)
            sentence_nll = sentence_nll_gpu.cpu().tolist()
            # print(sentence_nll)

            output = sentence_likelihood = [-x for x in sentence_nll]
            if return_dict:
                targets_mask_cpu = targets_mask.sum(dim=-1).cpu().tolist()
                output = {
                    'score': sentence_likelihood,
                    'step_score': (-target_tokens_nll).cpu().tolist(),
                    'num_subtokens': targets_mask_cpu
                    # 'input_tensors': {
                    #     x: v.cpu().tolist()
                    #     for x, v in batch_encoding.items()
                    # }
                }

                # compute ppl
                ppl = torch.exp(sentence_nll_gpu / targets_mask.sum(dim=-1)).cpu().tolist()
                output['ppl'] = ppl

            return output


class PretrainedLMWorker(mp.Process):
    def __init__(self, args: SimpleNamespace, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

    def run(self):
        print('init LM....')
        self.lm = PretrainedLM(self.args.model_name, self.device)

        assert hasattr(self, '_input_queue')

        while True:
            payload = self._input_queue.get()
            if payload == '_STOP_':
                self._result_queue.put('_STOP_')
                break

            payload: List[Tuple]
            sentences = [x[1] for x in payload]

            sentences_likelihood = self.lm.get_sentence_likelihood(sentences)

            results = []
            for idx, (example_id, sentence) in enumerate(payload):
                ll = sentences_likelihood[idx]
                results.append((example_id, ll))

            self._result_queue.put(results)

        print('Worker quit...')

    def register(self, input_queue: mp.Queue, result_queue: mp.Queue):
        self._input_queue = input_queue
        self._result_queue = result_queue


class Reader(mp.Process):
    def __init__(self, args: SimpleNamespace, input_queue: mp.Queue):
        super().__init__()

        self.args = args
        self._input_queue = input_queue

    def run(self):
        dataset = Path(self.args.dataset)
        batch_size = self.args.batch_size
        assert batch_size > 0

        batch = []
        for idx, line in enumerate(dataset.open()):
            if line.startswith('{'):
                data = json.loads(line)
                utterance = data['can']
            else:
                data = line.strip().split('\t')
                utterance = data[0]

            batch.append((idx, utterance))
            if len(batch) == batch_size:
                self._input_queue.put(batch)
                batch = []

        if batch:
            self._input_queue.put(batch)
            batch = []

        for _ in range(self.args.num_workers):
            self._input_queue.put('_STOP_')

        time.sleep(1)
        print('finished loading all data, reader stopped.')


class ScoringTask:
    def __init__(self, args: SimpleNamespace):
        self.workers = []
        for i in range(args.num_workers):
            device_name = f'cuda:{i}' if args.cuda else 'cpu'
            device = torch.device(device_name)
            print(device.type)

            worker = PretrainedLMWorker(args, device)
            self.workers.append(worker)

        self.input_queue = mp.Queue(maxsize=5000)
        self.result_queue = mp.Queue()

        for worker in self.workers:
            worker.register(self.input_queue, self.result_queue)

        self.reader = Reader(args, self.input_queue)
        self.total_num_examples = len(Path(args.dataset).open().readlines())

        self.args = args

    def run(self):
        self.reader.daemon = True
        self.reader.start()
        for worker in self.workers:
            worker.daemon = True
            worker.start()

        output_file = Path(self.args.output)
        if output_file.exists():
            os.system(f'rm {output_file}')

        f_output = output_file.open('w')

        num_stopped_workers = 0
        prog_bar = tqdm(desc='Decoding', total=self.total_num_examples)
        while True:
            result = self.result_queue.get()
            if result == '_STOP_':
                num_stopped_workers += 1
                if num_stopped_workers == len(self.workers):
                    print('all workers have stopped')
                    break
                else:
                    continue

            for example_id, ll in result:
                f_output.write(f'{example_id}\t{ll}\n')

            prog_bar.update(len(result))

        f_output.close()
        for worker in self.workers:
            worker.join()
        self.reader.join()
        print('Task finished!')


def test():
    lm = PretrainedLM('data/runs/gpt', device=torch.device('cpu'))
    nlls = lm.get_sentence_likelihood(
        [
            'papers that are coauthored by Dan Klein and Tom Mitchell',
            'papers written by Dan Klein and Tom Mitchell',
            'what are the papers that have Dan Klein and Tom Mitchell as co-authors',
            'What papers have Dan Klein and Tom Mitchell written ?',
            'what papers have Dan Klein and Tom Mitchell written?',
            'what papers have Dan Klein and Tom Mitchell written'
            #'NIPS 2018 paper',
            #'2018 NIPS paper',
            #'NIPS 2018 machine learning paper',
            #'2018 NIPS machine learning paper',
            #'machine learning NIPS 2018 paper',
            # 'Wikipedia articles',
            #'I love you.'
            # 'deep learning paper published in 2018'
        ],
        append_eos=False,
        return_dict=True
    )

    print(nlls)


def main():
    mp.set_start_method('spawn')

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=Path, required=True)
    arg_parser.add_argument('--output', type=Path, required=True)
    arg_parser.add_argument('--batch-size', type=int, default=10)
    arg_parser.add_argument('--model-name', type=str, default='gpt2')
    arg_parser.add_argument('--num-workers', type=int, default=4)
    arg_parser.add_argument('--cuda', action='store_true')

    args = arg_parser.parse_args()

    task = ScoringTask(args)
    task.run()


if __name__ == '__main__':
    # test()
    main()
