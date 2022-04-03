import argparse
import copy
import json
import os
import gc
import random
import re
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout

import torch
import wandb
import uuid
from argparse import ArgumentParser
from dataclasses import field, dataclass
from pathlib import Path
from typing import Optional, List, Dict, Union, cast, TypeVar
from allennlp.common.registrable import Registrable
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_module_and_submodules
from allennlp.models import load_archive
from allennlp.predictors import Predictor

from grammar_generation.program_utils import normalize_program_string
from nsp.metrics.denotation_accuracy import DenotationAccuracy
from nsp.metrics.denotation_accuracy_proxy import DenotationAccuracyProxy
from nsp.models.seq2seq_with_copy import Seq2SeqModelWithCopy
from nsp.metrics.token_sequence_accuracy import TokenSequenceAccuracy
from nsp.dataset_readers.seq2seq_with_copy_reader import SequenceToSequenceModelWithCopyReader

from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from sh import allennlp, cp, python

from common.config_dataclass import ParaphraseIdentificationModelConfig, ParaphraserConfig, ParserConfig
from common.utils import load_jsonl_file, write_jsonl_file
from paraphrase.generate_paraphrase import is_valid_paraphrased_example_according_to_parser_model
from paraphrase.paraphrase_identification import (
    train_and_infer as train_paraphrase_identification_model_and_infer,
    train_and_infer_on_paraphrase_tree,
    train_and_infer_on_paraphrase_tree_cross_validation,
    generate_parser_dataset_using_paraphrase_prediction_results, infer_paraphrase_oracle_experiment
)
from paraphrase.paraphrase_tree import ParaphraseTree, Node
from paraphrase.paraphrase_pruner import ParaphrasePruner, ParaphraseIdentificationModelPruner, ParserScoreParaphrasePruner
from paraphrase.sim.sim_utils import load_sim_model


DEBUG = os.getenv('DEBUG', False)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def clear_memory():
    print('Clear up GPU memory...')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('Memory Summary:')
        print(torch.cuda.memory_summary())
    gc.collect()


def evaluate(
    decode_results: List[Dict]
):
    all_hyps = []
    all_targets = []
    variables = []
    indices = []
    match_list = []
    for decode_result in decode_results:
        # assert instance['metadata']['source_tokens'] == decode_result['metadata']['source_tokens']

        hyps = [
            hyp['tokens']
            for hyp
            in decode_result['predictions']
            if len(hyp['tokens']) > 0
        ]

        target = decode_result['metadata']['target_tokens']

        if hyps:
            target_seq = ' '.join(target)
            top_hyp_seq = ' '.join(hyps[0])

            is_match = False
            try:
                is_match = normalize_program_string(target_seq) == normalize_program_string(top_hyp_seq)
            except Exception as e:
                print(f'Warning: error in parsing {top_hyp_seq}')

            match_list.append(is_match)

            if hyps[0] == target:
                print(hyps[0])

        all_hyps.append(hyps)
        all_targets.append(target)
        variables.append(decode_result['metadata']['variables'])
        indices.append(decode_result['metadata']['index'])

    metric = (
        DenotationAccuracyProxy(
            os.environ.get('EVALUATION_SERVER_ADDR', 'http://localhost:8081/')
        )
    )

    metric(all_hyps, all_targets, variables, indices)
    result = metric.get_metric(reset=True)

    seq_acc = sum(match_list) / len(decode_results)
    result['seq_acc'] = seq_acc

    return result


def perform_evaluation(
    model_path: Path,
    dataset_path: Path,
    decode_file_path: Optional[Path] = None
) -> Dict:
    print(f'Start evaluation using {dataset_path}....')
    decode_file_path = decode_file_path or (model_path.parent / dataset_path.with_suffix('.decode.jsonl').name)

    decode_run = allennlp(
        'predict',
        '--silent',
        '--use-dataset-reader',
        '--batch-size', 32,
        '--cuda-device', 0 if torch.cuda.is_available() else -1,
        '--include-package', 'nsp',
        '--output-file', decode_file_path,
        model_path,
        dataset_path,
        _out=sys.stdout,
        _err=sys.stderr
    )

    sys.stdout.flush()
    sys.stdout.flush()

    decode_result = load_jsonl_file(decode_file_path)
    t1 = time.time()
    print('Perform evaluation...')
    eval_result = evaluate(decode_result)
    t2 = time.time()

    eval_result['time'] = t2 - t1
    eval_result['decode_file'] = str(decode_file_path)

    return eval_result


def train_parser(
    iter_idx: int,
    train_file: Path,
    dev_file: Optional[Path],
    eval_file: Optional[Union[Path, List[Path]]],
    batch_size: int,
    work_dir: Path,
    seed: int,
    max_epoch: int,
    dropout: float,
    validation_metric: str = 'seq_acc',
    logical_form_data_field: str = 'lf',
    only_use_parser_filtered_paraphrase_example: bool = False,
    #use_canonical_example: bool = True,
    tgt_stopword_loss_weight: float = None,
    validation_after_epoch: int = 0,
    do_eval: bool = True,
    run_data: List[Dict] = None,
    config_file: Optional[Path] = None,
    from_pretrained: Optional[Path] = None,
    patience: int = 20,
    logger=None,
    **kwargs
):
    assert work_dir.exists()
    config_file = config_file or Path('configs/config_bert.json')
    assert config_file.exists()

    print('Start training parser.')

    extra_config = {
        'pytorch_seed': seed * 10,
        'random_seed': seed * 10 + 1,
        'numpy_seed': seed * 10 + 2,
        'train_data_path': train_file,
        'validation_data_path': dev_file,
        # 'test_data_path': test_file,
        'dataset_reader': {
            'logical_form_data_field': logical_form_data_field,
            'only_use_parser_filtered_example': only_use_parser_filtered_paraphrase_example,
            #'use_canonical_example': use_canonical_example
        },
        'model': {
            'validation_metric': validation_metric,
            'tgt_stopword_loss_weight': tgt_stopword_loss_weight,
            'decoder_dropout': dropout,
            'denotation_metric': {
                'executor_addr': os.getenv('EVALUATION_SERVER_ADDR', 'http://localhost:8081/')
            }
        },
        'trainer': {
            'validation_after_epoch': validation_after_epoch,
            'optimizer': {
                'lr': 3e-5,
            },
            'cuda_device': 0 if torch.cuda.is_available() else -1,
            'validation_metric': '-ppl' if validation_metric == 'ppl' else '+' + validation_metric
        }
        #'evaluate_on_test': True
    }

    if from_pretrained:
        from_pretrained = Path(from_pretrained)
        assert from_pretrained.exists()

        extra_config['model']['initializer'] = {
            'regexes':
            [
                [
                    '.*',
                    {
                        "type": "pretrained",
                        "weights_file_path": str(from_pretrained),
                        "parameter_name_overrides": {}
                    }
                ]
            ]
        }

        extra_config['vocabulary'] = {
            "type": "from_files",
            "directory": str(from_pretrained)
        }
    else:
        extra_config['vocabulary'] = {
            'min_count': {'source_tokens': 2, 'target_tokens': 1}
        }

    # if not only_use_parser_filtered_paraphrase_example:
    num_train_examples = len(open(extra_config["train_data_path"]).readlines())
    # else:
    #     train_examples = [
    #         json.loads(line)
    #         for line
    #         in open(extra_config["train_data_path"])
    #     ]
    #     filtered_examples = [
    #         e
    #         for e
    #         in train_examples
    #         if e.get('is_accepted_by_parser')
    #     ]
    #     num_train_examples = len(filtered_examples)

    max_epoch = 1 if DEBUG else max_epoch
    # if k in {100, 200}:
    #     max_epoch = 50
    #     patience = 20

    gradient_accumulation_steps = (batch_size // 32) if batch_size > 32 else 1
    # assert batch_size % 32 == 0
    total_train_steps = num_train_examples // batch_size * max_epoch
    num_steps_per_epoch = num_train_examples // batch_size

    extra_config.update({
        'data_loader': {
            'batch_sampler': {
                'batch_size': batch_size // gradient_accumulation_steps
            }
        }
    })
    extra_config['trainer']['num_epochs'] = max_epoch
    extra_config['trainer']['patience'] = patience
    extra_config['trainer']['num_gradient_accumulation_steps'] = gradient_accumulation_steps
    extra_config['trainer'].setdefault('learning_rate_scheduler', {})['warmup_steps'] = int(total_train_steps * 0.1)
    # extra_config['trainer'].setdefault('learning_rate_scheduler', {})['total_steps'] = total_train_steps

    extra_config_string = json.dumps(extra_config, default=str)

    print('Parser training with extra config:')
    print(json.dumps(extra_config_string, indent=2, default=str))
    sys.stdout.flush()
    sys.stderr.flush()

    # train_run = allennlp(
    #     'train',
    #     'configs/config_bert.json',
    #     '-s', work_dir,
    #     '--include-package', 'nsp',
    #     '-o', f"{extra_config_string}",
    #     # _out=sys.stdout,
    #     _err=sys.stderr
    # )

    if (work_dir / 'model.tar.gz').exists():
        print(f'[Iter {iter_idx}] WARNING: parer model already exists. Skipping training...')
    else:
        t1 = time.time()
        with (work_dir / 'stdout').open('w') as f:
            # with redirect_stdout(f):
                model = train_model_from_file(
                    parameter_filename=str(config_file),
                    serialization_dir=work_dir,
                    include_package=['nsp'],
                    overrides=extra_config_string,
                    file_friendly_logging=True
                )

                del model
                print(f'[Iter {iter_idx}] Clear up memory after training the parser...')
                clear_memory()

        t2 = time.time()
        if wandb.run:
            wandb.log({'iter': iter_idx, 'parser_train_time': t2 - t1})

    output_dict = {}

    sys.stdout.flush()
    sys.stderr.flush()

    if do_eval:
        eval_dataset_file = eval_file if not DEBUG and eval_file is not None else train_file
        decode_file = work_dir / eval_dataset_file.with_suffix('.decode.jsonl').name
        eval_result_file = decode_file.with_suffix('.eval_result')

        if eval_result_file.exists():
            print(f'[Iter {iter_idx}] WARNING: evaluation result already exists. Skipping evaluation...')
            eval_result = json.load(eval_result_file.open())
        else:
            eval_result = perform_evaluation(work_dir / 'model.tar.gz', eval_dataset_file)

            if wandb.run:
                wandb.log({'iter': iter_idx, 'evaluation_time': eval_result['time']})

            json.dump(eval_result, eval_result_file.open('w'))

        denotation_acc = eval_result['den_acc']
        print(eval_result)
        output_dict['dev_acc'] = denotation_acc
        history_eval_result = [x['dev_acc'] for x in run_data]
        print(history_eval_result)

        if wandb.run:
            wandb.log({'iter': iter_idx, 'denotation_acc': denotation_acc})
            wandb.log({'iter': iter_idx, 'seq_acc': eval_result['seq_acc']})
            cur_best = max(history_eval_result) if history_eval_result else 0.
            cur_best = max(cur_best, denotation_acc)
            wandb.run.summary['denotation_acc'] = cur_best

        sys.stdout.flush()
        sys.stderr.flush()

    return output_dict


def run_paraphraser(
    iter_idx: int,
    seed_dataset_file: Path,
    paraphraser_path: List[Path],
    output_file: Path,
    work_dir: Path,
    beam_size: int = 10,
    include_statement: bool = False,
    include_question: bool = True,
    parser_model_file: Path = None,
    sampling: bool = False,
    sampling_topp: float = 0.,
    include_source_examples: bool = False,
    heuristic_deduplicate: bool = False,
    parser_allowed_rank_in_beam: int = 1,
    batch_size: int = 64,
    parser_batch_size: int = 64,
    lm_scorer=None,
    paraphrase_tree: Optional[ParaphraseTree] = None,
    extra_config_string: str = '',
    **kwargs
):
    from paraphrase.generate_paraphrase import main as paraphraser_main

    config_dict = dict(
        input_dataset_file=seed_dataset_file,
        output_file=output_file,
        model_path=paraphraser_path,
        beam_size=beam_size,
        parser_batch_size=parser_batch_size,
        batch_size=batch_size,
        constrained_decoding=False,
        include_statement=include_statement,
        include_question=include_question,
        include_source_examples=include_source_examples,
        do_not_purge_beam=True,
        only_keep_parser_filtered_examples=False,
        sim_model_path=Path('paraphrase/sim/sim.pt'),
        lm_scorer=lm_scorer,
        heuristic_deduplicate=heuristic_deduplicate
    )

    if parser_model_file:
        config_dict['parser_model_file'] = parser_model_file
        config_dict['parser_allowed_rank_in_beam'] = parser_allowed_rank_in_beam

    if sampling:
        assert extra_config_string == ''
        config_dict['paraphraser_arg_string'] = json.dumps(dict(
            sampling=True,
            sampling_topp=sampling_topp
        ))
    else:
        config_dict['paraphraser_arg_string'] = extra_config_string

    # sys.stdout.flush()
    print(f'Paraphraser config:\n{json.dumps(config_dict, indent=2, default=str)}')
    # sys.stdout.flush()
    # sys.stderr.flush()

    t1 = time.time()
    result = paraphraser_main(**config_dict, paraphrase_tree=paraphrase_tree)
    t2 = time.time()
    if wandb.run:
        wandb.log({'iter': iter_idx, 'paraphraser_run_time': t2 - t1})
    # sys.stdout.flush()
    # sys.stderr.flush()

    return result


@torch.no_grad()
def label_examples_using_parser(examples: List[Dict], parser_model_file: Path, batch_size: int = 64):
    model_archive = load_archive(parser_model_file)
    if torch.cuda.is_available():
        print('Loading parser to cuda device.')
        model_archive.model.cuda()
        model_archive.model.eval()

    print('Set beam size to 1')
    parser_predictor = Predictor.from_archive(model_archive)
    cast(Seq2SeqModelWithCopy, parser_predictor._model)._beam_search.beam_size = 1
    cast(Seq2SeqModelWithCopy, parser_predictor._model)._validation_metric = None

    t1 = time.time()
    is_valid_paraphrased_example_according_to_parser_model(
        parser_predictor, examples, batch_size=batch_size)
    t2 = time.time()
    print(f'{t2 - t1:.1f}s took to label {len(examples)} examples')

    del model_archive
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return examples


def load_iter_status(file_path: Path) -> Dict:
    status_dict = json.load(file_path.open())
    key: str
    for key in list(status_dict.keys()):
        if key.endswith('path') or key.endswith('file'):
            status_dict[key] = Path(status_dict[key])

    return status_dict


def run(
    iter_idx: int,
    parser_train_file: Path,
    parser_dev_file: Optional[Path],
    parser_eval_file: Optional[Path],
    train_paraphrase_tree: ParaphraseTree,
    dev_paraphrase_tree: ParaphraseTree,
    work_dir: Path,
    parser_config: ParserConfig,
    paraphraser_config: ParaphraserConfig,
    paraphrase_identification_model_config: ParaphraseIdentificationModelConfig,
    paraphraser_model_file: List[Path],
    cumulative_data: List[Dict],
    prev_model_file: Path = None,
    seed: int = 0,
):
    output_dict = dict()

    print(f'[Iter {iter_idx}] clear CUDA memory at the beginning...')
    clear_memory()

    if iter_idx == 0 and not prev_model_file:
        # initial round, train only using the synthetic data
        print('Initial round, train the initial parser')
        parser_output_dict = train_parser(
            iter_idx,
            parser_train_file,
            parser_dev_file,
            parser_eval_file,
            work_dir=work_dir,
            seed=seed,
            run_data=cumulative_data,
            **parser_config.to_dict()
        )

        # initialize the paraphrase tree
        train_examples = load_jsonl_file(parser_train_file)
        train_paraphrase_tree.update_with_examples(train_examples)

        dev_examples = []
        if parser_dev_file:
            dev_examples = load_jsonl_file(parser_dev_file)
            dev_paraphrase_tree.update_with_examples(dev_examples)

        for example in train_examples + dev_examples:
            example['iter_idx'] = iter_idx

        return {
            'parser_train_file': parser_train_file,
            'paraphrased_train_file': None,
            'parser_dev_file': parser_dev_file,
            'paraphrased_dev_file': None,
            'is_canonical_train_run': True,
            **parser_output_dict
        }
    else:
        if (work_dir / 'iter_status.json').exists():
            print(f'[Iter {iter_idx}] It seems this iter is already finished. I am going to skip it...')
            print(f'[Iter {iter_idx}] Loading paraphrase trees...')
            train_paraphrase_tree.update_with_examples(load_jsonl_file(work_dir / 'paraphrase_tree.train.jsonl'))
            dev_paraphrase_tree.update_with_examples(load_jsonl_file(work_dir / 'paraphrase_tree.dev.jsonl'))

            output_dict = load_iter_status(work_dir / 'iter_status.json')
            return output_dict

        # call the paraphraser to paraphrase data
        paraphrased_dataset = {}

        splits = ['train']
        if parser_dev_file and paraphraser_config.paraphrase_dev_set:
            splits.append('dev')

        for split in splits:
            paraphrase_tree = (
                train_paraphrase_tree
                if split == 'train'
                else dev_paraphrase_tree
            )

            if iter_idx == 1:
                seed_file = cumulative_data[0][f'parser_{split}_file']
            else:
                seed_file: Path = cumulative_data[iter_idx - 1][f'paraphrased_{split}_file']
                if paraphraser_config.seed_file_type == 'original':
                    pass
                elif paraphraser_config.seed_file_type == 'filtered':
                    seed_file = seed_file.with_suffix('.filtered.jsonl')
                elif paraphraser_config.seed_file_type == 'parser_input':
                    seed_file = seed_file.with_suffix('.parser_input.jsonl')
                elif paraphraser_config.seed_file_type == 'leaf_nodes':
                    # collect valid leaf nodes
                    cand_examples = []
                    for level in range(1, paraphrase_tree.depth):
                        cand_examples.extend([
                            e.value for e in paraphrase_tree.get_nodes_by_level(level)
                            if len(e.children) == 0 and e.value['is_valid_paraphrase']
                        ])

                    seed_file = work_dir / f'{split}.paraphrased.iter{iter_idx}.source_examples.jsonl'
                    write_jsonl_file(cand_examples, seed_file)
                else:
                    raise ValueError('Unknown seed file type: ' + paraphraser_config.seed_file_type)

            output_file_name = f'{split}.paraphrased.iter{iter_idx}.jsonl'
            output_file = work_dir / output_file_name
            print(f'Paraphraser Input File: {seed_file}')

            if split == 'train' and paraphraser_config.do_not_filter_train_set_using_parser:
                parser_model_file = None
            else:
                parser_model_file = prev_model_file

            if not output_file.exists():
                paraphraser_config = copy.copy(paraphraser_config)
                if iter_idx >= 3:
                    paraphraser_config.batch_size = paraphraser_config.batch_size // 2

                paraphrased_examples, paraphraser_log_dict = run_paraphraser(
                    iter_idx,
                    seed_file,
                    paraphraser_path=paraphraser_model_file,
                    output_file=output_file,
                    parser_model_file=parser_model_file,
                    work_dir=work_dir,
                    paraphrase_tree=paraphrase_tree,
                    **paraphraser_config.to_dict()
                )
            else:
                print(
                    f'[Iter {iter_idx}] WARNING: Skipping the current iteration of paraphrasing. '
                    f'Loading results from existing file: [{output_file}]'
                )
                paraphraser_log_dict = {}
                paraphrased_examples = load_jsonl_file(output_file)

            for example in paraphrased_examples:
                example['iter_idx'] = iter_idx

            if not paraphraser_config.do_not_filter_train_set_using_parser:
                for example in paraphrased_examples:
                    example.setdefault('accepted_by_parser_in_iter', {})[iter_idx - 1] = example['is_accepted_by_parser']

            paraphrase_tree.update_with_examples(paraphrased_examples)

            paraphrased_dataset[split] = {
                'file': output_file,  # type: Path
                'examples': paraphrased_examples,
                'paraphraser_log': paraphraser_log_dict
            }

            if not DEBUG:
                filtered_examples = [
                    e
                    for e
                    in paraphrased_examples
                    if (
                        e.get('is_accepted_by_parser') or
                        (
                            paraphraser_config.include_source_examples and
                            not e.get('is_paraphrase')
                        ) or
                        (
                            paraphraser_config.filter_example_by_sim_score_threshold and
                            e.get('sim_score', 0.) > paraphraser_config.filter_example_by_sim_score_threshold
                        )
                    )
                ]
            else:
                filtered_examples = paraphrased_examples

            filtered_dataset_file = output_file.with_suffix('.filtered.jsonl')
            write_jsonl_file(filtered_examples, filtered_dataset_file)

            paraphrased_dataset[split]['paraphraser_filtered_file'] = filtered_dataset_file
            paraphrased_dataset[split]['paraphraser_filtered_examples'] = filtered_examples

            clear_memory()

        # run co-training of the paraphrase identification model
        if paraphrase_identification_model_config.enabled:
            print(f'[Iter {iter_idx}] Run Paraphrase Identification Model...')
            pi_model_args = copy.copy(paraphrase_identification_model_config)
            if iter_idx > 1:
                pi_model_args.model_name_or_path = cumulative_data[-1]['paraphrase_identification_model_path']

            from_iterations = []
            if parser_config.use_cumulated_datasets:
                from_iterations.extend(list(range(1, iter_idx)))
            from_iterations.append(iter_idx)

            clear_memory()

            # update the paraphrase tree
            t1 = time.time()
            if paraphrase_identification_model_config.oracle_experiment:
                train_program_to_natural_utterances: Dict = json.load(Path('data/train_scholar_nat.all_derives_scholar_6.578d6fef.pruned_lf.k10000.template_split.normalized_d1cd1398.jsonl.program_to_natural_utterances.json').open())
                dev_program_to_natural_utterances: Dict = json.load(Path('data/dev_scholar_nat.all_derives_scholar_6.578d6fef.pruned_lf.k10000.template_split.normalized_d1cd1398.jsonl.program_to_natural_utterances.json').open())

                for prog, entry in dev_program_to_natural_utterances.items():
                    train_program_to_natural_utterances.setdefault(prog, {'natural': [], 'canonical': []})['natural'].extend(entry['natural'])
                    train_program_to_natural_utterances[prog]['canonical'].extend(entry['canonical'])

                for prog in list(train_program_to_natural_utterances):
                    train_program_to_natural_utterances[prog]['natural'] = sorted(list(set(train_program_to_natural_utterances[prog]['natural'])))

                pi_output_dict = infer_paraphrase_oracle_experiment(
                    train_paraphrase_tree,
                    dev_paraphrase_tree,
                    train_program_to_natural_utterances,
                    args=pi_model_args
                )

                with (work_dir / 'pi_model_debug.predictions.json').open('w') as f:
                    json.dump(pi_output_dict, f, indent=2)
            else:
                if paraphrase_identification_model_config.num_folds > 1:
                    pi_output_dict = train_and_infer_on_paraphrase_tree_cross_validation(
                        train_paraphrase_tree,
                        dev_paraphrase_tree=dev_paraphrase_tree,
                        args=pi_model_args,
                        work_dir=work_dir,
                        from_iterations=from_iterations,
                        num_folds=paraphrase_identification_model_config.num_folds
                    )
                else:
                    pi_output_dict = train_and_infer_on_paraphrase_tree(
                        train_paraphrase_tree,
                        dev_paraphrase_tree=dev_paraphrase_tree,
                        args=pi_model_args,
                        work_dir=work_dir,
                        current_iter_index=iter_idx,
                        from_iterations=from_iterations,
                    )
            t2 = time.time()
            if wandb.run:
                wandb.log({'iter': iter_idx, 'paraphrase_id_model_run_time': t2 - t1})

            pi_predictions = pi_output_dict['predictions']

            # gather candidate train examples
            cand_train_examples = []
            cand_dev_examples = []
            for _iter_idx in from_iterations:
                examples = train_paraphrase_tree.get_examples_by_level(_iter_idx)
                cand_train_examples.extend(examples)

                examples = dev_paraphrase_tree.get_examples_by_level(_iter_idx)
                cand_dev_examples.extend(examples)

            # generate the final training set
            for example in cand_train_examples + cand_dev_examples:
                if not paraphrase_identification_model_config.only_use_parser_accepted_examples and not paraphrase_identification_model_config.oracle_experiment:
                    assert example['idx'] in pi_predictions, f"{example['idx']} is not included in predictions!"

                label = pi_predictions.get(example['idx'], False)
                example['is_valid_paraphrase'] = label
                example.setdefault('paraphrase_identification_labels', {})[iter_idx] = label

            parser_train_examples = [
                e for e in cand_train_examples
                if e['is_valid_paraphrase']
            ]

            parser_dev_examples = [
                e for e in cand_dev_examples
                if e['is_valid_paraphrase']
            ]

            if paraphrase_identification_model_config.oracle_experiment:
                if paraphrase_identification_model_config.oracle_experiment_dev_split_ratio > 0:
                    assert parser_config.use_cumulated_datasets
                    dev_ratio = paraphrase_identification_model_config.oracle_experiment_dev_split_ratio

                    rng = random.Random(1234)

                    parser_train_examples = []
                    parser_dev_examples = []

                    for root_example in train_paraphrase_tree.get_examples_by_level(0):
                        child_examples = list(train_paraphrase_tree.get_descent_examples(root_example['idx']))
                        child_examples = [
                            e for e in child_examples
                            if e['is_valid_paraphrase']
                        ]

                        train_examples_num = int(len(child_examples) * (1.0 - dev_ratio))
                        rng.shuffle(child_examples)

                        train_examples_ = child_examples[:train_examples_num]
                        dev_examples_ = child_examples[train_examples_num:]

                        parser_train_examples.extend(train_examples_)
                        parser_dev_examples.extend(dev_examples_)

            clear_memory()

            if paraphrase_identification_model_config.use_pruner:
                print(f'[Iter {iter_idx}] Prune paraphrase results using a pruner')

                with torch.no_grad():
                    sim_model = load_sim_model()
                    paraphrase_pruner_cls = Registrable.by_name(paraphrase_identification_model_config.pruner_name)
                    if paraphrase_identification_model_config.pruner_name == 'parser_score_pruner':
                        prev_parser_archive = load_archive(
                            prev_model_file,
                            cuda_device=0 if torch.cuda.is_available() else -1
                        )
                        pruner_arg = dict(
                            parser_archive=prev_parser_archive,
                            paraphrase_tree=train_paraphrase_tree,
                            sim_model=sim_model,
                            K=paraphrase_identification_model_config.pruner_nbr_num,
                            index_all_descendants=paraphrase_identification_model_config.pruner_index_all_descendants
                        )

                    else:
                        pruner_arg = dict(
                            paraphrase_tree=train_paraphrase_tree,
                            paraphrase_identification_model_path=pi_output_dict['model_path'],
                            sim_model=sim_model,
                            K=paraphrase_identification_model_config.pruner_nbr_num,
                            index_all_descendants=paraphrase_identification_model_config.pruner_index_all_descendants
                        )

                    t1 = time.time()
                    paraphrase_pruner = paraphrase_pruner_cls(**pruner_arg)
                    parser_train_examples = paraphrase_pruner.prune(parser_train_examples, iter_idx=iter_idx)

                    #if paraphrase_identification_model_config.process_dev_data:
                    pruner_arg['paraphrase_tree'] = dev_paraphrase_tree
                    paraphrase_pruner = paraphrase_pruner_cls(**pruner_arg)
                    parser_dev_examples = paraphrase_pruner.prune(parser_dev_examples, iter_idx=iter_idx)

                    t2 = time.time()
                    if wandb.run:
                        wandb.log({'iter': iter_idx, 'paraphrase_pruner_run_time': t2 - t1})

                    del paraphrase_pruner
                    del sim_model

                    if paraphrase_identification_model_config.pruner_name == 'parser_score_pruner':
                        del prev_parser_archive

            # apply the transitive rule: if the parent is not a valid paraphrase, then all its children are not
            def transitive_rule(node_: Node):
                if not node_.value['is_valid_paraphrase']:
                    for example_ in paraphrase_tree.get_descent_examples(node_.value['idx']):
                        example_['is_valid_paraphrase'] = False
                else:
                    for child_node_ in node_.children:
                        transitive_rule(child_node_)

            for paraphrase_tree in [train_paraphrase_tree, dev_paraphrase_tree]:
                for node in paraphrase_tree.get_nodes_by_level(0):
                    for child_node in node.children:
                        transitive_rule(child_node)

            # re-generate filtered dataset
            filtered_train_examples = [
                example
                for example
                in train_paraphrase_tree.get_examples_by_level(iter_idx)
                if example['is_valid_paraphrase']
            ]

            filtered_dev_examples = [
                example
                for example
                in dev_paraphrase_tree.get_examples_by_level(iter_idx)
                if example['is_valid_paraphrase']
            ]

            paraphrased_dataset['train']['paraphraser_filtered_examples'] = filtered_train_examples
            filtered_dataset_file = paraphrased_dataset['train']['file'].with_suffix('.filtered.jsonl')
            write_jsonl_file(filtered_train_examples, filtered_dataset_file)

            paraphrased_dataset['dev']['paraphraser_filtered_examples'] = filtered_dev_examples
            filtered_dataset_file = paraphrased_dataset['dev']['file'].with_suffix('.filtered.jsonl')
            write_jsonl_file(filtered_dev_examples, filtered_dataset_file)

            transitive_filtered_parser_train_examples = [
                example
                for example
                in parser_train_examples
                if example['is_valid_paraphrase']
            ]

            transitive_filtered_parser_dev_examples = [
                example
                for example
                in parser_dev_examples
                if example['is_valid_paraphrase']
            ]

            if wandb.run:
                wandb.log({'iter': iter_idx, f'transitive_filtered_ratio_train': len(transitive_filtered_parser_train_examples) / len(parser_train_examples)})
                wandb.log({'iter': iter_idx, f'transitive_filtered_ratio_dev': len(transitive_filtered_parser_dev_examples) / len(parser_dev_examples)})

            paraphrased_dataset['train']['parser_examples'] = transitive_filtered_parser_train_examples
            paraphrased_dataset['dev']['parser_examples'] = transitive_filtered_parser_dev_examples

            output_dict['paraphrase_identification_model_path'] = pi_output_dict['model_path']

            print(f'[Iter {iter_idx}] Clean up memory after running the pruner')
            clear_memory()

        if not paraphrase_identification_model_config.enabled or not paraphrase_identification_model_config.process_dev_data:
            splits = ['train', 'dev']
            if paraphrase_identification_model_config.enabled:
                print('[Model] Do not run paraphrase ID model on dev data.')
                splits = ['dev']

            if not paraphraser_config.paraphrase_dev_set or not parser_dev_file:
                splits.remove('dev')

            for split in splits:
                paraphrase_tree = train_paraphrase_tree if split == 'train' else dev_paraphrase_tree

                # mark examples with the parser's acceptance labels.
                for example in paraphrase_tree.get_examples_by_level(iter_idx):
                    example['is_valid_paraphrase'] = example['is_accepted_by_parser']

                parser_examples = []
                if parser_config.use_cumulated_datasets:
                    print('Using previous examples')
                    parser_examples = []

                    if parser_config.use_canonical_example:
                        parser_examples.extend(paraphrase_tree.get_examples_by_level(0))

                    for i in range(1, iter_idx):
                        parser_examples.extend([
                            e
                            for e
                            in paraphrase_tree.get_examples_by_level(i)
                            if e['is_valid_paraphrase']
                        ])

                parser_examples.extend([
                    e
                    for e
                    in paraphrase_tree.get_examples_by_level(iter_idx)
                    if e['is_valid_paraphrase']
                ])

                paraphrased_dataset[split]['parser_examples'] = parser_examples

        if not paraphraser_config.paraphrase_dev_set:
            print('Do not paraphrase dev samples, use original dev set.')
            paraphrased_dataset['dev'] = {
                'file': work_dir / f'dev.paraphrased.iter{iter_idx}.jsonl'
            }

            paraphrased_dataset['dev']['parser_examples'] = dev_paraphrase_tree.get_examples_by_level(0) if parser_dev_file else []
            paraphrased_dataset['dev']['examples'] = paraphrased_dataset['dev']['paraphraser_filtered_examples'] = paraphrased_dataset['dev']['parser_examples']

        train_examples = paraphrased_dataset['train']['parser_examples']
        dev_examples = []
        if parser_dev_file:
            dev_examples = paraphrased_dataset['dev']['parser_examples']

        if wandb.run:
            wandb.log({'iter': iter_idx, f'num_parser_train_examples': len(train_examples)})
            wandb.log({'iter': iter_idx, f'num_parser_dev_examples': len(dev_examples)})
            wandb.log({'iter': iter_idx, f'num_paraphrased_train_examples': len(paraphrased_dataset['train']['examples'])})
            wandb.log({'iter': iter_idx, f'num_paraphrased_filtered_train_examples': len(paraphrased_dataset['train']['paraphraser_filtered_examples'])})

            for key, val in paraphrased_dataset['train']['paraphraser_log'].items():
                wandb.log({f'iter': iter_idx, f'paraphraser_train_{key}': val})

            if parser_dev_file:
                wandb.log({'iter': iter_idx, f'num_paraphrased_dev_examples': len(paraphrased_dataset['dev']['examples'])})
                wandb.log({'iter': iter_idx, f'num_paraphrased_filtered_dev_examples': len(paraphrased_dataset['dev']['paraphraser_filtered_examples'])})

                if 'paraphraser_log' in paraphrased_dataset['dev']:
                    for key, val in paraphrased_dataset['dev']['paraphraser_log'].items():
                        wandb.log({f'iter': iter_idx, f'paraphraser_dev_{key}': val})

        parser_train_file = paraphrased_dataset['train']['file'].with_suffix('.parser_input.jsonl')
        write_jsonl_file(train_examples, parser_train_file)

        if parser_dev_file:
            parser_dev_file = paraphrased_dataset['dev']['file'].with_suffix('.parser_input.jsonl')
            write_jsonl_file(dev_examples, parser_dev_file)

        # duplicate this because I want to get visualizations asap
        train_paraphrase_tree.save_to(work_dir / 'paraphrase_tree.train.jsonl')
        dev_paraphrase_tree.save_to(work_dir / 'paraphrase_tree.dev.jsonl')

        iter_parser_config = copy.copy(parser_config)
        iter_parser_config.only_use_parser_filtered_paraphrase_example = False
        parser_config_dict = iter_parser_config.to_dict()
        # train a new parser
        parser_output_dict = train_parser(
            iter_idx,
            parser_train_file,
            parser_dev_file,
            parser_eval_file,
            work_dir=work_dir,
            seed=seed,
            run_data=cumulative_data,
            **parser_config_dict
        )
        output_dict.update(parser_output_dict)

    if parser_config.rerun_prediction:
        print(f'[Iter {iter_idx}] Predicting examples on the paraphrase tree with the latest model.')
        t1 = time.time()

        for paraphrase_tree in [train_paraphrase_tree, dev_paraphrase_tree]:
            parser_examples = []
            for i in range(1, iter_idx + 1):
                parser_examples.extend([
                    e
                    for e
                    in paraphrase_tree.get_examples_by_level(i)
                    if iter_idx not in e.get('accepted_by_parser_in_iter', {})
                ])

            if parser_examples:
                parser_examples = label_examples_using_parser(parser_examples, work_dir / 'model.tar.gz')
                print(f'[Iter {iter_idx}] Clear up memory after prediction.')

                for example in parser_examples:
                    example_: Dict = paraphrase_tree.get_example_by_id(example['idx'])
                    example_.setdefault('accepted_by_parser_in_iter', {})[iter_idx] = example['is_accepted_by_parser']
                    example_['is_accepted_by_parser'] = example['is_accepted_by_parser']
                    example_['is_valid_paraphrase'] = example['is_accepted_by_parser']

                clear_memory()

        t2 = time.time()
        if wandb.run:
            wandb.log({'iter': iter_idx, 'parser_prediction_time': t2 - t1})

    train_paraphrase_tree.save_to(work_dir / 'paraphrase_tree.train.jsonl')
    dev_paraphrase_tree.save_to(work_dir / 'paraphrase_tree.dev.jsonl')

    output_dict.update(
        {
            'parser_train_file': parser_train_file,
            'paraphrased_train_file': paraphrased_dataset['train']['file'],
            'parser_dev_file': parser_dev_file,
            'paraphrased_dev_file': paraphrased_dataset['dev']['file'] if parser_dev_file else None
        }
    )

    json.dump(output_dict, (work_dir / 'iter_status.json').open('w'), indent=2, default=str)

    return output_dict


def main(
    seed: int,
    seed_train_file: Path,
    seed_dev_file: Optional[Path],
    eval_file: Path,
    test_file: Optional[Path],
    work_dir: Path,
    paraphraser_model_file: List[Path],
    start_iteration: int = 0,
    end_iteration: int = 4,
    seed_model_file: Optional[Path] = None,
    parser_batch_size: List[int] = [64],
    parser_max_epoch: List[int] = [30],
    parser_patience: int = 20,
    parser_decoder_dropout: float = 0.2,
    parser_validation_metric: str = 'seq_acc',
    parser_validation_after_epoch: int = 0,
    parser_use_canonical_example: bool = True,
    parser_use_cumulated_datasets: bool = False,
    parser_rerun_prediction: bool = False,
    parser_logical_form_data_field: str = 'lf',
    parser_target_stopword_loss_weight: Optional[float] = None,
    parser_config_file: Optional[Path] = None,
    parser_from_pretrained: Optional[Path] = None,
    eval_parser: bool = True,
    only_use_parser_filtered_paraphrase_example: bool = True,
    paraphraser_batch_size: int = 64,
    paraphraser_parser_batch_size: int = 64,
    paraphraser_beam_size: Union[int, List[int]] = 10,
    paraphraser_sampling: List[bool] = None,
    paraphraser_sampling_topp: List[float] = None,
    paraphraser_include_statement: bool = False,
    paraphraser_include_question: bool = False,
    paraphraser_include_source_examples: List[bool] = None,
    paraphraser_lm_scorer: str = None,
    paraphraser_seed_file_type: str = 'filtered',
    paraphraser_do_not_filter_train_set_using_parser: bool = False,
    paraphrase_dev_set: bool = True,
    paraphraser_heuristic_deduplicate: bool = False,
    paraphraser_parser_allowed_rank_in_beam: int = 1,
    paraphraser_extra_config: str = '',
    paraphrase_identification_model_config: ParaphraseIdentificationModelConfig = None,
    filter_example_by_sim_score_threshold: Optional[float] = None,
    restore_from: Optional[Path] = None,
    **kwargs
):
    print('Running multi-iteration model with args:')
    config_dict = locals()
    config_dict['paraphrase_identification_model_config'] = paraphrase_identification_model_config.to_dict()
    if 'config_dict' in config_dict:
        del config_dict['config_dict']

    config_dict_log = json.dumps(config_dict, default=str, indent=2)
    print(config_dict_log)

    if not DEBUG:
        wandb.init(project='data_efficient_parsing', job_type='iterative_train')
        wandb.config.update(json.loads(config_dict_log))
        wandb.config.work_dir = work_dir

        print(f'wandb run name: {wandb.run.name}')
        print(f'wandb run ID: {wandb.run.id}')
        print(f'wandb run path: {wandb.run.path}')

        config_dict['wandb_run_path'] = wandb.run.path
        config_dict['wandb_run_id'] = wandb.run.id
        config_dict['wandb_run_name'] = wandb.run.name

    if not work_dir.exists():
        work_dir.mkdir(parents=True)

    with open(work_dir / 'config.json', 'w') as f:
        f.write(json.dumps(config_dict, default=str, indent=2))

    # if DEBUG:
    #     shutil.rmtree(work_dir)

    import_module_and_submodules('nsp')

    paraphrase_identification_model_config = paraphrase_identification_model_config or ParaphraseIdentificationModelConfig()

    if seed_model_file:
        assert seed_model_file.exists()

    if parser_from_pretrained:
        assert parser_from_pretrained.exists()

    if paraphraser_model_file:
        for path in paraphraser_model_file:
            assert path.exists()

    if restore_from is not None:
        print(f'Restore training from {restore_from} @ Iteration {start_iteration}')
        print(f'Will ignore seed_train_file and seed_dev_file')

    assert seed_train_file.exists()
    if seed_dev_file is not None:
        assert seed_dev_file.exists()
    assert eval_file.exists()
    if test_file:
        assert test_file.exists()

    num_iter = end_iteration - start_iteration
    assert num_iter > 0

    if len(parser_batch_size) < num_iter:
        parser_batch_size.extend([parser_batch_size[-1]] * num_iter)

    if len(parser_max_epoch) < num_iter:
        parser_max_epoch.extend([parser_max_epoch[-1]] * num_iter)

    if paraphraser_sampling is None:
        assert paraphraser_sampling_topp is None

        paraphraser_sampling = [False] * num_iter
        paraphraser_sampling_topp = [0.] * num_iter
    else:
        assert len(paraphraser_sampling) == num_iter
        assert len(paraphraser_sampling_topp) == num_iter

    paraphraser_include_source_examples = paraphraser_include_source_examples or [False] * num_iter
    if len(paraphraser_include_source_examples) < num_iter:
        paraphraser_include_source_examples.extend([False] * (num_iter - len(paraphraser_include_source_examples)))

    if isinstance(paraphraser_beam_size, int):
        paraphraser_beam_size = [paraphraser_beam_size] * num_iter
    elif len(paraphraser_beam_size) < num_iter:
        paraphraser_beam_size.extend([paraphraser_beam_size[-1]] * (num_iter - len(paraphraser_beam_size)))

    print(f'Work dir is: {work_dir}')
    work_dir.mkdir(exist_ok=True, parents=True)
    print(f'Seed training file: {seed_train_file}')
    print(f'Seed training file: {seed_dev_file}')

    # if seed_model_file:
    #     assert start_iteration > 0

    cur_model_file = seed_model_file
    cur_train_file = cur_dev_file = None
    cumulative_data = []

    train_paraphrase_tree = ParaphraseTree()
    dev_paraphrase_tree = ParaphraseTree()

    if start_iteration > 0:
        assert start_iteration == 1
        cumulative_data.append({
            'parser_train_file': seed_train_file,
            'parser_dev_file': seed_dev_file,
            'dev_acc': 0.
        })

        # load examples
        seed_train_examples = load_jsonl_file(seed_train_file)
        train_paraphrase_tree.update_with_examples(seed_train_examples)

        seed_dev_examples = []
        if seed_dev_file:
            seed_dev_examples = load_jsonl_file(seed_dev_file)
            dev_paraphrase_tree.update_with_examples(seed_dev_examples)

        for example in seed_train_examples + seed_dev_examples:
            example['iter_idx'] = 0

    for iter_idx in range(start_iteration, end_iteration):
        print(f'Iteration {iter_idx}')
        sys.stdout.flush()

        iter_work_dir = work_dir / f'round1_iter{iter_idx}'
        iter_work_dir.mkdir(parents=True, exist_ok=True)

        if iter_idx == start_iteration:
            cur_train_file = iter_work_dir / seed_train_file.with_suffix(f'.iter{start_iteration}.jsonl').name
            cp(seed_train_file, cur_train_file)

            cur_dev_file = None
            if seed_dev_file:
                cur_dev_file = iter_work_dir / seed_dev_file.with_suffix(f'.iter{start_iteration}.jsonl').name
                cp(seed_dev_file, cur_dev_file)

        parser_config = ParserConfig(
            batch_size=parser_batch_size[iter_idx - start_iteration],
            max_epoch=parser_max_epoch[iter_idx - start_iteration],
            patience=parser_patience,
            validation_metric=parser_validation_metric,
            only_use_parser_filtered_paraphrase_example=only_use_parser_filtered_paraphrase_example,
            use_canonical_example=(
                True
                if iter_idx == 0 and not seed_model_file
                else parser_use_canonical_example
            ),
            do_eval=eval_parser,
            use_cumulated_datasets=parser_use_cumulated_datasets,
            rerun_prediction=parser_rerun_prediction,
            tgt_stopword_loss_weight=parser_target_stopword_loss_weight,
            validation_after_epoch=parser_validation_after_epoch,
            dropout=parser_decoder_dropout,
            logical_form_data_field=parser_logical_form_data_field,
            config_file=parser_config_file,
            from_pretrained=parser_from_pretrained
        )

        paraphraser_config = ParaphraserConfig(
            batch_size=paraphraser_batch_size,
            parser_batch_size=paraphraser_parser_batch_size,
            beam_size=paraphraser_beam_size[iter_idx - start_iteration],
            include_statement=paraphraser_include_statement,
            include_question=paraphraser_include_question,
            sampling=paraphraser_sampling[iter_idx - start_iteration],
            sampling_topp=paraphraser_sampling_topp[iter_idx - start_iteration],
            include_source_examples=paraphraser_include_source_examples[iter_idx - start_iteration],
            filter_example_by_sim_score_threshold=filter_example_by_sim_score_threshold,
            paraphrase_dev_set=paraphrase_dev_set,
            seed_file_type=paraphraser_seed_file_type,
            heuristic_deduplicate=paraphraser_heuristic_deduplicate,
            do_not_filter_train_set_using_parser=paraphraser_do_not_filter_train_set_using_parser,
            parser_allowed_rank_in_beam=paraphraser_parser_allowed_rank_in_beam,
            lm_scorer=paraphraser_lm_scorer,
            extra_config_string=paraphraser_extra_config
        )

        result_dict = run(
            iter_idx,
            cur_train_file,
            cur_dev_file,
            eval_file,
            parser_config=parser_config,
            paraphraser_config=paraphraser_config,
            paraphrase_identification_model_config=paraphrase_identification_model_config,
            train_paraphrase_tree=train_paraphrase_tree,
            dev_paraphrase_tree=dev_paraphrase_tree,
            work_dir=iter_work_dir,
            prev_model_file=cur_model_file,
            paraphraser_model_file=paraphraser_model_file,
            cumulative_data=cumulative_data,
            seed=seed,
        )

        cur_model_file = iter_work_dir / 'model.tar.gz'
        cur_train_file = result_dict['parser_train_file']
        cur_dev_file = result_dict['parser_dev_file']
        cumulative_data.append(result_dict)

    if test_file:
        assert cur_model_file.exists()
        print(f'Perform testing on {test_file}....')
        test_result = perform_evaluation(
            cur_model_file,
            test_file
        )
        test_denotation_acc = test_result["den_acc"]
        print(f'test accuracy: {test_denotation_acc}')
        json.dump(
            test_result,
            (work_dir / (test_file.name + '.eval_result')).open('w')
        )

        json.dump(
            test_result,
            (cur_model_file.parent / (test_file.name + '.eval_result')).open('w')
        )

        if wandb.run:
            wandb.run.summary['test_denotation_acc'] = test_denotation_acc


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--seed', type=int, default=0)
    arg_parser.add_argument('--seed-train-file', type=Path, required=True)
    arg_parser.add_argument('--seed-dev-file', type=Path, required=False, default=None)
    arg_parser.add_argument('--eval-file', type=Path, required=False, default=Path('data/test_scholar.json'))
    arg_parser.add_argument('--test-file', type=Path, required=False, default=None)
    arg_parser.add_argument('--seed-model-file', type=Path)

    arg_parser.add_argument('--work-dir', type=Path, required=True)

    # arg_parser.add_argument('--paraphraser', choices=['bart', 'genienlp'], default='bart')
    arg_parser.add_argument('--paraphraser-model-file', type=Path, nargs='+', default=[Path('data/paraphrase/checkpoints-maxtok512-updatefreq2-iter5000-warmup500/checkpoint_best.pt')])
    arg_parser.add_argument('--paraphraser-beam-size', nargs='+', type=int, default=[10])
    arg_parser.add_argument('--paraphraser-batch-size', type=int, default=64)
    arg_parser.add_argument('--paraphraser-parser-batch-size', type=int, default=64)
    arg_parser.add_argument('--paraphraser-lm-scorer', type=str, default=None)
    arg_parser.add_argument('--paraphraser-sampling', nargs='+', type=str2bool)
    arg_parser.add_argument('--paraphraser-sampling-topp', nargs='+', type=float)
    arg_parser.add_argument('--paraphraser-include-source-examples', nargs='+', type=str2bool)
    arg_parser.add_argument('--paraphraser-include-statement', action='store_true')
    arg_parser.add_argument('--paraphraser-include-question', action='store_true')
    arg_parser.add_argument('--paraphraser-extra-config', type=str, default='')
    arg_parser.add_argument('--paraphraser-seed-file-type', default='filtered', choices=['original', 'filtered', 'parser_input', 'leaf_nodes'])
    arg_parser.add_argument('--no-paraphrase-dev-set', action='store_false', dest='paraphrase_dev_set', default=True)
    arg_parser.add_argument('--paraphraser-heuristic-deduplicate', action='store_true')
    arg_parser.add_argument('--paraphraser-do-not-filter-train-set-using-parser', action='store_true')
    arg_parser.add_argument('--paraphraser-parser-allowed-rank-in-beam', type=int, default=1)

    arg_parser.add_argument('--parser-batch-size', type=int, nargs='+', default=[32])
    arg_parser.add_argument('--parser-max-epoch', type=int, nargs='+', default=[30])
    arg_parser.add_argument('--parser-patience', type=int, default=20)
    arg_parser.add_argument('--parser-decoder-dropout', type=float, default=0.2)
    arg_parser.add_argument('--parser-target-stopword-loss-weight', type=float, default=None)
    arg_parser.add_argument('--parser-validation-metric', type=str, default='seq_acc', choices=['seq_acc', 'ppl', 'denotation_acc'])
    arg_parser.add_argument('--parser-validation-after-epoch', type=int, default=0)
    arg_parser.add_argument('--no-eval-parser', dest='eval_parser', default=True, action='store_false')
    arg_parser.add_argument('--parser-not-use-canonical-example', action='store_false', dest='parser_use_canonical_example', default=True)
    arg_parser.add_argument('--parser-use-cumulated-datasets', action='store_true')
    arg_parser.add_argument('--parser-rerun-prediction', action='store_true')
    arg_parser.add_argument('--parser-logical-form-data-field', default='lf', type=str)
    arg_parser.add_argument('--parser-config-file', type=Path, default=None)
    arg_parser.add_argument('--parser-from-pretrained', type=Path, default=None)

    arg_parser.add_argument('--use-paraphrase-identification-model', action='store_true')
    arg_parser.add_argument('--paraphrase-identification-model-oracle-exp', action='store_true')
    arg_parser.add_argument('--paraphrase-identification-model-oracle-exp-label-strategy', type=str, default='majority')
    arg_parser.add_argument('--paraphrase-identification-model-oracle-exp-dev-split-ratio', type=float, default=0.)
    arg_parser.add_argument('--paraphrase-identification-model-not-run-on-dev-data', dest='paraphrase_identification_model_run_on_dev_data', default=True, action='store_false')
    arg_parser.add_argument('--paraphrase-identification-model-only-use-parser-accepted-examples', action='store_true')
    arg_parser.add_argument('--paraphrase-identification-model-lr', type=float, default=2e-5)
    arg_parser.add_argument('--paraphrase-identification-model-use-pruner', action='store_true')
    arg_parser.add_argument('--paraphrase-identification-model-pruner-nbr-num', type=int, default=5)
    arg_parser.add_argument('--paraphrase-identification-model-pruner-name', type=str, default=None)
    arg_parser.add_argument('--paraphrase-identification-model-pruner-index-all-descendants', action='store_true')
    arg_parser.add_argument('--paraphrase-identification-model-use-model-labeled-positive-examples', action='store_true')
    arg_parser.add_argument('--paraphrase-identification-model-name-or-path', type=str)
    arg_parser.add_argument('--paraphrase-identification-model-num-folds', type=int, default=1)
    arg_parser.add_argument('--paraphrase-identification-model-sample-size', type=int, default=None)
    arg_parser.add_argument('--paraphrase-identification-model-batch-size', type=int, default=32)
    arg_parser.add_argument('--paraphrase-identification-model-gradient-accumulation-steps', type=int, default=1)
    arg_parser.add_argument('--paraphrase-identification-model-no-sample-positive-example-by-sim-score', action='store_false', dest='pi_sample_positive_example_by_sim_score', default=True)
    arg_parser.add_argument('--paraphrase-identification-model-inference-only', action='store_true')

    arg_parser.add_argument('--start-iteration', type=int, default=0)
    arg_parser.add_argument('--end-iteration', type=int, default=4)

    arg_parser.add_argument('--filter-example-by-sim-score-threshold', type=float, default=None)

    args = arg_parser.parse_args()

    paraphrase_identification_model_config = ParaphraseIdentificationModelConfig()
    if args.use_paraphrase_identification_model:
        paraphrase_identification_model_config.enabled = True
        paraphrase_identification_model_config.oracle_experiment = args.paraphrase_identification_model_oracle_exp
        paraphrase_identification_model_config.oracle_experiment_label_strategy = args.paraphrase_identification_model_oracle_exp_label_strategy
        paraphrase_identification_model_config.oracle_experiment_dev_split_ratio = args.paraphrase_identification_model_oracle_exp_dev_split_ratio
        paraphrase_identification_model_config.process_dev_data = args.paraphrase_identification_model_run_on_dev_data
        paraphrase_identification_model_config.only_use_parser_accepted_examples = args.paraphrase_identification_model_only_use_parser_accepted_examples
        paraphrase_identification_model_config.lr = args.paraphrase_identification_model_lr
        paraphrase_identification_model_config.gradient_accumulation_steps = args.paraphrase_identification_model_gradient_accumulation_steps
        paraphrase_identification_model_config.use_pruner = args.paraphrase_identification_model_use_pruner
        paraphrase_identification_model_config.pruner_name = args.paraphrase_identification_model_pruner_name
        paraphrase_identification_model_config.pruner_nbr_num = args.paraphrase_identification_model_pruner_nbr_num
        paraphrase_identification_model_config.pruner_index_all_descendants = args.paraphrase_identification_model_pruner_index_all_descendants
        paraphrase_identification_model_config.use_model_labeled_positive_examples = args.paraphrase_identification_model_use_model_labeled_positive_examples
        paraphrase_identification_model_config.num_folds = args.paraphrase_identification_model_num_folds
        paraphrase_identification_model_config.batch_size = args.paraphrase_identification_model_batch_size
        paraphrase_identification_model_config.sample_size = args.paraphrase_identification_model_sample_size
        paraphrase_identification_model_config.sample_negative_example_by_sim_score = True
        paraphrase_identification_model_config.sample_positive_example_by_sim_score = args.pi_sample_positive_example_by_sim_score
        paraphrase_identification_model_config.model_name_or_path = args.paraphrase_identification_model_name_or_path
        paraphrase_identification_model_config.inference_only = args.paraphrase_identification_model_inference_only

    main(**vars(args), paraphrase_identification_model_config=paraphrase_identification_model_config)
