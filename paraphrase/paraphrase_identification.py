import copy
import csv
import gc
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Optional, Union, Any

import torch
from sh import python

import numpy as np

from common.config_dataclass import ParaphraseIdentificationModelConfig
from paraphrase.paraphrase_tree import ParaphraseTree
# from paraphrase.hf_paraphrase_identification import main as run_pi_model
from paraphrase.utils import get_paraphrase_tree


def get_label(label):
    return 1 if label else 0


def load_jsonl(
    file_path: Path
):
    return [
        json.loads(line)
        for line
        in Path(file_path).open()
    ]


def dump_samples(
    samples: List[Dict],
    file_path: Path,
    is_test: bool = False
):
    with Path(file_path).open('w') as f:
        fieldnames = ['id', 'sentence1', 'sentence2']
        if not is_test:
            fieldnames.append('label')

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames, delimiter=','
        )

        writer.writeheader()
        for sample in samples:
            output_dict = dict(
                id=sample["idx"],
                sentence1=sample["sent_1"],
                sentence2=sample["sent_2"],
            )
            if not is_test:
                output_dict['label'] = sample.get("label", 0)

            writer.writerow(output_dict)


def generate_dataset(
    paraphrase_file: Path,
    output_prefix: Path,
    sample_size: Optional[int] = 1000,
    sample_negative_example_by_sim_score: bool = False,
    sample_positive_example_by_sim_score: bool = False,
    split_train_dev=True
):
    paraphrase_examples = load_jsonl(paraphrase_file)
    paraphrase_tree = get_paraphrase_tree(paraphrase_examples)

    original_example_idx_list = sorted(list(paraphrase_tree))

    np.random.seed(1234)
    np.random.shuffle(original_example_idx_list)

    samples = []
    example_count = 0
    example_ptr = 0
    if sample_size is None:
        sample_size = len(paraphrase_tree)

    while example_count < sample_size and example_ptr < len(paraphrase_tree):
        example = paraphrase_tree[original_example_idx_list[example_ptr]]
        paraphrases = list(map(lambda e: e[1], sorted(example['children'].items(), key=lambda e: e[0])))

        accepted_examples = [
            e for e in paraphrases
            if e['is_accepted_by_parser']
        ]

        rejected_examples = [
            e for e in paraphrases
            if not e['is_accepted_by_parser']
        ]

        if accepted_examples and rejected_examples:
            if sample_positive_example_by_sim_score:
                paraphrased_example = sorted(accepted_examples, key=lambda e: e['sim_score'])[0]
            else:
                paraphrased_example = np.random.choice(accepted_examples)

            sample = dict(
                idx=paraphrased_example['idx'],
                sent_1=example['can'],
                sent_2=paraphrased_example['can'],
                label=get_label(paraphrased_example['is_accepted_by_parser'])
            )
            samples.append(sample)

            if sample_negative_example_by_sim_score:
                paraphrased_example = sorted(rejected_examples, key=lambda e: e['sim_score'])[0]
            else:
                paraphrased_example = np.random.choice(rejected_examples)

            sample = dict(
                idx=paraphrased_example['idx'],
                sent_1=example['can'],
                sent_2=paraphrased_example['can'],
                label=get_label(paraphrased_example['is_accepted_by_parser'])
            )
            samples.append(sample)

            example_count += 1

        example_ptr += 1

    all_samples = []
    for example in paraphrase_tree.values():
        for paraphrased_example in example['children'].values():
            sample = dict(
                idx=paraphrased_example['idx'],
                sent_1=example['can'],
                sent_2=paraphrased_example['can'],
                label=get_label(paraphrased_example['is_accepted_by_parser'])
            )

            all_samples.append(sample)

    if split_train_dev:
        dev_ratio = 0.2
        num_train = int(len(samples) * (1 - dev_ratio))
        train_samples = samples[:num_train]
        dev_samples = samples[num_train:]

        dump_samples(train_samples, output_prefix.with_name(output_prefix.name + '.sampled.train.csv'))
        dump_samples(dev_samples, output_prefix.with_name(output_prefix.name + '.sampled.dev.csv'))
    else:
        dump_samples(samples, output_prefix.with_name(output_prefix.name + '.sampled.csv'))

    dump_samples(all_samples, output_prefix.with_name(output_prefix.name + '.all.csv'), is_test=True)

    return all_samples


def generate_dataset_from_paraphrase_tree(
    paraphrase_tree: ParaphraseTree,
    sample_size_per_iteration: Optional[int] = None,
    sample_negative_example_by_sim_score: bool = False,
    sample_positive_example_by_sim_score: bool = False,
    sample_positive_example_by_parser_score: bool = False,
    sample_negative_example_by_parser_score: bool = False,
    use_model_labeled_positive_examples: bool = False,
    from_iterations: List[int] = None,
    only_use_parser_accepted_examples: bool = False,
    run_identifier: str = '0'
):
    from_iterations = from_iterations or list(range(paraphrase_tree.depth))

    np.random.seed(1234)

    samples = []
    for iter_idx in from_iterations:
        parent_level = iter_idx - 1
        parent_examples = paraphrase_tree.get_examples_by_level(parent_level)
        iter_sample_size = sample_size_per_iteration if sample_size_per_iteration else 999999999999999

        iter_samples = []
        example_ptr = 0

        np.random.shuffle(parent_examples)

        while len(iter_samples) < iter_sample_size and example_ptr < len(parent_examples):
            example = parent_examples[example_ptr]
            node = paraphrase_tree.get_node_by_id(example['idx'])
            paraphrases: List[Dict] = [child.value for child in node.children]

            if (
                not paraphrases or
                only_use_parser_accepted_examples and not any(e['is_accepted_by_parser'] for e in paraphrases)
            ):
                example_ptr += 1
                continue

            # at the beginning of training, the paraphrases are not labeled by the paraphrase ID model.
            # We instead use the parser's predictions as labels
            is_paraphrase_already_labeled = 'is_valid_paraphrase' in paraphrases[0]
            if is_paraphrase_already_labeled and use_model_labeled_positive_examples:
                accepted_examples = [
                    e for e in paraphrases
                    if (
                        e['is_valid_paraphrase'] and
                        all(
                            e_['is_valid_paraphrase']
                            for e_
                            in paraphrase_tree.get_ancester_examples(e['idx'])
                            if 'paraphrase' in str(e_['idx'])
                        )
                    )
                ]

                rejected_examples = [
                    e for e in paraphrases
                    if (
                        not e['is_valid_paraphrase']
                    )
                ]
            else:
                # use parser's predictions as labels
                accepted_examples = [
                    e for e in paraphrases
                    if (
                        e['is_accepted_by_parser']
                    )
                ]

                rejected_examples = [
                    e for e in paraphrases
                    if (
                        not e['is_accepted_by_parser']
                    )
                ]

            if accepted_examples and rejected_examples:
                if sample_positive_example_by_sim_score:
                    paraphrased_example = sorted(accepted_examples, key=lambda e: e['sim_score'])[0]
                elif sample_positive_example_by_parser_score:
                    paraphrased_example = sorted(accepted_examples, key=lambda e: e['parser_score'])[0]
                else:
                    paraphrased_example = np.random.choice(accepted_examples)

                paraphrased_example.setdefault('pid_model_metadata', []).append(f'{run_identifier}:T')
                sample = dict(
                    idx=f'{paraphrased_example["idx"]}-pos',
                    sent_1=example['can'],
                    sent_2=paraphrased_example['can'],
                    label=get_label(True)
                )
                iter_samples.append(sample)

                # if 'root-leaf' in data_augmentation_pattern:


                if sample_negative_example_by_sim_score:
                    paraphrased_example = sorted(rejected_examples, key=lambda e: e['sim_score'])[0]
                elif sample_negative_example_by_parser_score:
                    paraphrased_example = sorted(rejected_examples, key=lambda e: e['parser_score'])[0]
                else:
                    paraphrased_example = np.random.choice(rejected_examples)

                paraphrased_example.setdefault('pid_model_metadata', []).append(f'{run_identifier}:F')
                sample = dict(
                    idx=f'{paraphrased_example["idx"]}-neg',
                    sent_1=example['can'],
                    sent_2=paraphrased_example['can'],
                    label=get_label(False)
                )
                iter_samples.append(sample)

            example_ptr += 1

        samples.extend(iter_samples)

    all_samples = []
    for level in range(paraphrase_tree.depth):
        parent_examples = paraphrase_tree.get_examples_by_level(level)
        for example in parent_examples:
            child_nodes = paraphrase_tree.get_node_by_id(example['idx']).children

            if only_use_parser_accepted_examples and not any(e.value['is_accepted_by_parser'] for e in child_nodes):
                continue

            for child_node in paraphrase_tree.get_node_by_id(example['idx']).children:
                paraphrased_example = child_node.value
                sample = dict(
                    idx=paraphrased_example['idx'],
                    sent_1=example['can'],
                    sent_2=paraphrased_example['can'],
                    label=get_label(paraphrased_example['is_accepted_by_parser'])
                )

                all_samples.append(sample)

    return {
        'all_examples_pi_model_data': all_samples,
        'pi_model_sampled_data': samples
    }


def load_paraphrase_identification_model_prediction(dataset_file: Path, prediction_file: Path, return_dict: bool = False) -> Dict[str, Union[bool, Dict]]:
    predictions = dict()
    with dataset_file.open() as f, prediction_file.open() as f_pred:
        reader = csv.DictReader(f)
        pred_reader = csv.DictReader(f_pred, dialect='excel-tab')
        for entry, pred in zip(reader, pred_reader):
            example_idx = entry['id']
            label = pred['prediction']
            prob = pred.get('prob')
            datum = {
                'label': int(label) == 1
            }
            if prob:
                prob = [float(x) for x in prob.split('|')]
                datum['prob'] = prob

            if example_idx not in predictions:
                print(f'WARNING: duplicate example idx [{example_idx}] in {prediction_file}', file=sys.stderr)
            if return_dict:
                predictions[example_idx] = datum
            else:
                predictions[example_idx] = datum['label']

    return predictions


def generate_training_data(
    paraphrase_file: Path,
    dataset_file: Path,
    prediction_file: Path,
    output_file: Path,
    include_parser_accepted_examples: bool = True
):
    predictions = load_paraphrase_identification_model_prediction(dataset_file, prediction_file)
    examples = load_jsonl(paraphrase_file)

    filtered_examples = [
        example
        for example
        in examples
        if predictions[example['idx']] or (example['is_accepted_by_parser'] if include_parser_accepted_examples else False)
    ]

    with output_file.open('w') as f:
        for example in filtered_examples:
            f.write(json.dumps(example) + '\n')


def generate_parser_dataset_using_paraphrase_prediction_results(
    examples: List[Dict],
    # predictions: Dict = None,
    include_parser_accepted_examples: bool = False
):
    filtered_examples = [
        example
        for example
        in examples
        if example['is_valid_paraphrase'] or (example['is_accepted_by_parser'] if include_parser_accepted_examples else False)
    ]

    return filtered_examples


def train_and_infer_on_paraphrase_tree_cross_validation(
    train_paraphrase_tree: ParaphraseTree,
    work_dir: Path,
    from_iterations: List[int],
    args: ParaphraseIdentificationModelConfig,
    dev_paraphrase_tree: Optional[ParaphraseTree] = None,
    num_folds: int = 10
):
    paraphrase_trees = [('train', train_paraphrase_tree)]
    paraphrase_trees.append(('dev', dev_paraphrase_tree))

    output_dicts = {}
    for split_name, paraphrase_tree in paraphrase_trees:
        output_dict = generate_dataset_from_paraphrase_tree(
            paraphrase_tree,
            from_iterations=from_iterations,
            sample_size_per_iteration=args.sample_size,
            sample_positive_example_by_sim_score=args.sample_positive_example_by_sim_score,
            sample_negative_example_by_sim_score=args.sample_negative_example_by_sim_score,
            use_model_labeled_positive_examples=args.use_model_labeled_positive_examples
        )

        output_dicts[split_name] = output_dict

    suffix = f'pi_{args.sample_size}_lr{args.lr}_ep{args.epoch}_ngsim{args.sample_negative_example_by_sim_score}'
    dev_data_file = work_dir / f'{suffix}.dev.sampled.csv'
    dump_samples(output_dicts['dev']['pi_model_sampled_data'], dev_data_file)

    train_samples = output_dicts['train']['pi_model_sampled_data']

    # group samples by there seed example id
    grouped_train_examples = OrderedDict()
    for sample in train_samples:
        root_idx = str(sample['idx']).partition('-')[0]
        grouped_train_examples.setdefault(root_idx, []).append(sample)

    grouped_train_examples_root_ids = list(grouped_train_examples.keys())
    grouped_train_examples = list(grouped_train_examples.values())
    num_grouped_examples_per_fold = len(grouped_train_examples) // num_folds

    all_train_example_root_ids = set(
        train_paraphrase_tree.get_root_idx(e['idx'])
        for e
        in output_dicts['train']['all_examples_pi_model_data']
    )

    # assert '1629024' in all_train_example_root_ids

    gathered_prediction_results = dict()
    for fold_idx in range(num_folds):
        fold_train_data_file = work_dir / f'{suffix}.train.fold_{fold_idx}.sampled.csv'
        fold_train_examples = (
            grouped_train_examples[:num_grouped_examples_per_fold * fold_idx]
        )
        if fold_idx < num_folds - 1:
            fold_train_examples += grouped_train_examples[num_grouped_examples_per_fold * (fold_idx + 1):]

        fold_train_examples = list(itertools.chain(*fold_train_examples))
        dump_samples(fold_train_examples, fold_train_data_file)

        held_out_train_sample_root_ids = all_train_example_root_ids - set(
            train_paraphrase_tree.get_root_idx(e['idx'])
            for e
            in fold_train_examples
        )

        # if '1629024' in held_out_train_sample_root_ids:
        #     print(f'Fold {fold_idx} has the tgt data point.')
        #
        # if '1629024' in set(
        #     train_paraphrase_tree.get_root_idx(e['idx'])
        #     for e
        #     in fold_train_examples
        # ):
        #     print(f'Fold {fold_idx} has the training data point.')

        head_out_train_samples = [
            sample
            for sample
            in output_dicts['train']['all_examples_pi_model_data']
            if train_paraphrase_tree.get_root_idx(sample['idx']) in held_out_train_sample_root_ids
        ]

        samples_to_infer = head_out_train_samples + output_dicts['dev']['all_examples_pi_model_data']
        samples_to_infer_data_file = work_dir / f'{suffix}.train.fold_{fold_idx}.to_infer.csv'
        dump_samples(samples_to_infer, samples_to_infer_data_file)

        model_name = f'{suffix}_model_fold{fold_idx}'

        if (
            (work_dir / model_name).exists() and
            (work_dir / model_name / 'pytorch_model.bin').exists() and
            (work_dir / model_name / 'test_results_None.txt').exists()
        ):
            print(f'WARNING: Model path: {work_dir / model_name} already exists, skipping this fold...')
        else:
            if (work_dir / model_name).exists():
                raise RuntimeError(f'Folder {work_dir / model_name} exists and non-empty')
            #     shutil.rmtree(work_dir / model_name)

            num_train_examples = len(fold_train_examples)
            num_total_steps = num_train_examples // args.batch_size // args.gradient_accumulation_steps * args.epoch
            num_warmup_steps = int(0.1 * num_total_steps)

            train_args = [
                '--model_name_or_path', str(args.model_name_or_path),
                '--do_train',
                '--do_eval',
                '--do_predict',
                '--per_device_train_batch_size', str(args.batch_size),
                '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
                '--per_device_eval_batch_size', str(args.batch_size),
                '--learning_rate', str(args.lr),
                '--warmup_steps', str(num_warmup_steps),
                '--num_train_epochs', str(args.epoch),
                '--output_dir', str(work_dir / model_name),
                '--train_file', str(fold_train_data_file),
                '--validation_file', str(dev_data_file),
                '--test_file', str(samples_to_infer_data_file),
                '--load_best_model_at_end',
                '--metric_for_best_model', 'accuracy',
                '--evaluation_strategy', 'epoch',
                '--save_total_limit', '1',
                '--save_steps', '99999999',   # a high save_steps so we don't save intermediate models.
                '--logging_dir', str(work_dir / model_name)
                #'--overwrite_output_dir'
            ]

            env = copy.copy(os.environ)

            python(
                '-m',
                'paraphrase.hf_paraphrase_identification',
                *train_args,
                _err=sys.stderr,
                _out=sys.stdout,
                _env=env
            )

            for ckpt_folder in (work_dir / model_name).glob('checkpoint-*'):
                print(f'Deleting {ckpt_folder}...')
                shutil.rmtree(ckpt_folder)

        # run_pi_model(train_args)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        prediction_results = load_paraphrase_identification_model_prediction(
            samples_to_infer_data_file,
            work_dir / model_name / 'test_results_None.txt'
        )

        with (samples_to_infer_data_file.with_suffix('.predictions.tsv')).open('w') as f:
            for key, val in prediction_results.items():
                f.write(f'{key}\t{1 if val else 0}\n')

        for idx, label in prediction_results.items():
            gathered_prediction_results.setdefault(idx, []).append(label)

    # reduction
    reduced_predictions = dict()
    for idx, labels in gathered_prediction_results.items():
        pos_count = sum(1 for x in labels if x)
        neg_count = sum(1 for x in labels if not x)

        if pos_count >= neg_count:
            label = True
        else:
            label = False

        reduced_predictions[idx] = label

    with (work_dir / f'{suffix}.to_infer.predictions.tsv').open('w') as f:
        for key, val in reduced_predictions.items():
            f.write(f'{key}\t{1 if val else 0}\n')

    # average model weights
    model_state_dict: Dict[str, torch.Tensor] = OrderedDict()

    with torch.no_grad():
        for fold_idx in range(num_folds):
            model_name = f'{suffix}_model_fold{fold_idx}'
            state_dict: Dict = torch.load(work_dir / model_name / 'pytorch_model.bin', map_location='cpu')
            tensor: torch.Tensor
            for key, tensor in state_dict.items():
                if key not in model_state_dict:
                    model_state_dict[key] = tensor
                elif tensor.dtype == torch.float32:
                    model_state_dict[key] = model_state_dict[key] + tensor
                else:
                    model_state_dict[key] = tensor

            del state_dict

        for key in model_state_dict:
            tensor = model_state_dict[key]
            if tensor.dtype == torch.float32:
                model_state_dict[key] = tensor / float(num_folds)

    avg_model_name = f'{suffix}_model'

    if (work_dir / avg_model_name).exists():
        print(f'WARNING: Model {work_dir / avg_model_name} already exists. Skipping...')
    else:
        (work_dir / avg_model_name).mkdir(parents=True)

        torch.save(model_state_dict, work_dir / avg_model_name / 'pytorch_model.bin')
        for file_name in [
            'config.json', 'training_args.bin',
            'merges.txt', 'special_tokens_map.json', 'tokenizer_config.json',
            'vocab.json'
        ]:
            shutil.copy(work_dir / f'{suffix}_model_fold0' / file_name, work_dir / avg_model_name / file_name)

    return {
        'predictions': reduced_predictions,
        'model_path': work_dir / avg_model_name
    }


def train_and_infer_on_paraphrase_tree(
    train_paraphrase_tree: ParaphraseTree,
    work_dir: Path,
    current_iter_index: int,  # starts from 1
    from_iterations: List[int],
    args: ParaphraseIdentificationModelConfig,
    dev_paraphrase_tree: Optional[ParaphraseTree] = None,
    overwrite: bool = False
) -> Dict:
    paraphrase_trees = [('train', train_paraphrase_tree)]
    if len(dev_paraphrase_tree) > 0:
        paraphrase_trees.append(('dev', dev_paraphrase_tree))

    suffix = f'pi_{args.sample_size}_lr{args.lr}_ep{args.epoch}_ngsim{args.sample_negative_example_by_sim_score}'
    all_samples = []

    output_dict = {}
    pi_train_datasets = {}
    num_train_examples = -1
    for split_name, paraphrase_tree in paraphrase_trees:
        output_dict = generate_dataset_from_paraphrase_tree(
            paraphrase_tree,
            from_iterations=from_iterations,
            sample_size_per_iteration=args.sample_size,
            sample_positive_example_by_sim_score=args.sample_positive_example_by_sim_score,
            sample_negative_example_by_sim_score=args.sample_negative_example_by_sim_score,
            use_model_labeled_positive_examples=args.use_model_labeled_positive_examples,
            only_use_parser_accepted_examples=args.only_use_parser_accepted_examples,
            run_identifier=work_dir.name.rpartition('_')[-1]
        )

        samples_file = work_dir / f'{suffix}.{split_name}.sampled.csv'
        pi_train_datasets[split_name] = samples_file
        dump_samples(output_dict['pi_model_sampled_data'], samples_file)
        all_samples.extend(output_dict['all_examples_pi_model_data'])

        if split_name == 'train':
            num_train_examples = len(output_dict['pi_model_sampled_data'])

    all_sample_file = work_dir / f'pi_data.{suffix}.csv'
    dump_samples(all_samples, all_sample_file)

    model_name = suffix + '_model'

    skip_training = False
    if (work_dir / model_name).exists() and (work_dir / model_name / 'config.json').exists():
        if overwrite:
            print(f'Warning: I am going to delete the folder: {work_dir / model_name}')
            shutil.rmtree(work_dir / model_name)
        else:
            print(f"WARNING: I am going to skip training for model {work_dir / model_name}...")
            skip_training = True
    else:
        try:
            print(f'Warning: I am going to delete the folder: {work_dir / model_name}')
            shutil.rmtree(work_dir / model_name)
        except: pass

    inference_only = args.inference_only

    if not skip_training:
        num_total_steps = num_train_examples // args.batch_size // args.gradient_accumulation_steps * args.epoch
        num_warmup_steps = int(0.1 * num_total_steps)

        train_args = [
            '--model_name_or_path', str(args.model_name_or_path),
            '--do_predict',
            '--max_seq_length', 64,
            '--per_device_train_batch_size', str(args.batch_size),
            '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
            '--per_device_eval_batch_size', str(args.batch_size * 2),
            '--learning_rate', str(args.lr),
            '--num_train_epochs', str(args.epoch),
            '--warmup_steps', str(num_warmup_steps),
            '--weight_decay', str(0.1),
            '--max_grad_norm', str(10.0),
            '--output_dir', str(work_dir / model_name),
            '--train_file', str(pi_train_datasets['train']),
            '--validation_file', str(pi_train_datasets['dev']),
            '--test_file', str(all_sample_file),
            '--load_best_model_at_end',
            '--metric_for_best_model', 'accuracy',
            '--evaluation_strategy', 'epoch',
            '--save_total_limit', '1',
            '--logging_dir', str(work_dir / model_name),
        ]

        if current_iter_index > 1 and not inference_only:
            print('[Paraphrase ID Model] train the paraphraser')
            train_args.extend(['--do_train', '--do_eval'])
        else:
            print('[Paraphrase ID Model] only perform evaluation')
            shutil.copytree(args.model_name_or_path, work_dir / model_name)

        # run_pi_model(train_args)

        env = copy.copy(os.environ)
        env['WANDB_MODE'] = 'dryrun'

        python(
            '-m',
            'paraphrase.hf_paraphrase_identification',
            *train_args,
            _err=sys.stderr,
            _out=sys.stdout,
            _env=env
        )

        for ckpt_folder in (work_dir / model_name).glob('checkpoint-*'):
            print(f'Deleting {ckpt_folder}...')
            shutil.rmtree(ckpt_folder)

    for ckpt_folder in (work_dir / model_name).glob('checkpoint-*'):
        print(f'Deleting {ckpt_folder}...')
        shutil.rmtree(ckpt_folder)

    # python(
    #     '/ocean/projects/dbs200003p/pcyin/Research/transformers/examples/text-classification/run_glue.py',
    #     '--model_name_or_path', model_path,
    #     '--do_train',
    #     '--do_eval',
    #     '--do_predict',
    #     '--per_device_train_batch_size', args.batch_size,
    #     '--learning_rate', args.lr,
    #     '--num_train_epochs', args.epoch,
    #     '--output_dir', work_dir / model_name,
    #     '--train_file', pi_train_datasets['train'],
    #     '--validation_file', pi_train_datasets['dev'],
    #     '--test_file', all_sample_file,
    #     '--load_best_model_at_end',
    #     '--metric_for_best_model', 'accuracy',
    #     '--evaluation_strategy', 'epoch',
    #     '--save_total_limit', 1,
    #     _err=sys.stderr,
    #     _out=sys.stdout
    # )

    prediction_results = load_paraphrase_identification_model_prediction(
        all_sample_file,
        work_dir / model_name / 'test_results_None.txt'
    )

    output_dict['predictions'] = prediction_results
    output_dict['model_path'] = work_dir / model_name

    return output_dict


def train_and_infer(
    args
) -> Dict:
    work_dir: Path = args.work_dir
    suffix = f'pi_{args.sample_size}_lr{args.lr}_ep{args.epoch}_ngsim{args.sample_negative_example_by_sim_score}'
    all_samples = []
    for paraphrase_example_file_name in [args.input_train_file_name, args.input_dev_file_name]:
        samples = generate_dataset(
            work_dir / paraphrase_example_file_name,
            output_prefix=(work_dir / paraphrase_example_file_name).with_suffix('.' + suffix),
            sample_size=args.sample_size,
            split_train_dev=False,
            sample_negative_example_by_sim_score=args.sample_negative_example_by_sim_score
        )
        all_samples.extend(samples)

    all_sample_file = work_dir / f'pi_data.{suffix}.csv'
    dump_samples(all_samples, all_sample_file)

    train_file = (work_dir / args.input_train_file_name).with_suffix('.' + suffix + '.sampled.csv')
    dev_file = (work_dir / args.input_dev_file_name).with_suffix('.' + suffix + '.sampled.csv')

    model_name = suffix + '_model'

    if (work_dir / model_name).exists():
        shutil.rmtree(work_dir / model_name)

    model_path = Path(args.model_name_or_path)
    python(
        '-m',
        'paraphrase.hf_paraphrase_identification'
        '--model_name_or_path', model_path,
        '--do_train',
        '--do_eval',
        '--do_predict',
        '--per_device_train_batch_size', args.batch_size,
        '--learning_rate', args.lr,
        '--num_train_epochs', args.epoch,
        '--output_dir', work_dir / model_name,
        '--train_file', train_file,
        '--validation_file', dev_file,
        '--test_file', all_sample_file,
        '--load_best_model_at_end',
        '--metric_for_best_model', 'accuracy',
        '--evaluation_strategy', 'epoch',
        '--save_total_limit', 1,
        _err=sys.stderr,
        _out=sys.stdout
    )

    output_dict = dict()
    for paraphrase_example_file_name in [args.input_train_file_name, args.input_dev_file_name]:
        output_file = (work_dir / paraphrase_example_file_name).with_suffix('.' + suffix + '.w_parser_acc.parser_input.jsonl')
        generate_training_data(
            paraphrase_file=work_dir / paraphrase_example_file_name,
            dataset_file=all_sample_file,
            prediction_file=work_dir / model_name / 'test_results_None.txt',
            output_file=output_file,
            include_parser_accepted_examples=True
        )

        output_dict[paraphrase_example_file_name] = {
            'with_parser_accepted_sample': output_file
        }

        output_file = (work_dir / paraphrase_example_file_name).with_suffix('.' + suffix + '.parser_input.jsonl')
        generate_training_data(
            paraphrase_file=work_dir / paraphrase_example_file_name,
            dataset_file=all_sample_file,
            prediction_file=work_dir / model_name / 'test_results_None.txt',
            output_file=output_file,
            include_parser_accepted_examples=False
        )

        output_dict[paraphrase_example_file_name]['without_parser_accepted_sample'] = output_file

    output_dict['model_path'] = work_dir / model_name

    return output_dict


def infer_paraphrase_oracle_experiment(
    train_paraphrase_tree: ParaphraseTree,
    dev_paraphrase_tree: ParaphraseTree,
    program_to_natural_utterances: Dict[str, Dict[str, List[str]]],
    args: ParaphraseIdentificationModelConfig
):
    from grammar_generation.generate_dataset_from_sampled_examples import PLACEHOLDER_ENTITY_MAP
    from grammar_generation.generate_oracle_dataset import normalize_program_string

    fb_entity_to_name = {}
    for key, value in PLACEHOLDER_ENTITY_MAP.items():
        if isinstance(value['entity'], str):
            fb_entity_to_name[value['entity']] = key

    def replace(string, vocab):
        for key, val in vocab.items():
            string = string.replace(key, fb_entity_to_name.get(val, val) if isinstance(val, str) else val[1])

        return string

    all_samples = []

    root_example_natural_utterances = {}

    for tag, paraphrase_tree in [
        ('train', train_paraphrase_tree),
        ('dev', dev_paraphrase_tree)
    ]:
        for level in range(paraphrase_tree.depth):
            parent_examples = paraphrase_tree.get_examples_by_level(level)
            for example in parent_examples:
                canonical_lf = normalize_program_string(example['lf'])
                if canonical_lf in program_to_natural_utterances:
                    natural_utterances = program_to_natural_utterances[canonical_lf]['natural']

                    if level == 0:
                        assert example['idx'] not in root_example_natural_utterances
                        root_example_natural_utterances[example['idx']] = natural_utterances

                    for child_node in paraphrase_tree.get_node_by_id(example['idx']).children:
                        paraphrased_example = child_node.value
                        for utt_idx, natural_utterance in enumerate(natural_utterances):
                            sample = dict(
                                idx=f'{paraphrased_example["idx"]}-#nat#-{utt_idx}',
                                sent_1=replace(paraphrased_example['nl'], paraphrased_example['variables']),
                                sent_2=replace(natural_utterance, paraphrased_example['variables']),
                                label=0
                            )

                            all_samples.append(sample)

    predictions = run_paraphrase_identification_model(
        all_samples,
        model_path=Path(args.model_name_or_path),
        return_dict=False
    )

    grouped_predictions = {}
    for idx, label in predictions.items():
        example_idx, _, nat_idx = idx.partition('-#nat#-')
        natural_utterance = root_example_natural_utterances[
            paraphrase_tree.get_root_idx(example_idx)
        ][int(nat_idx)]
        grouped_predictions.setdefault(example_idx, {})[int(nat_idx)] = {
            'label': label,
            'natural_utterance': natural_utterance
        }

    final_predictions = {}
    for example_idx, group in grouped_predictions.items():
        num_pos_label = sum(1 for x in group.values() if x['label'])
        num_neg_label = len(group) - num_pos_label

        if args.oracle_experiment_label_strategy == 'majority':
            label = num_pos_label > num_neg_label
        elif args.oracle_experiment_label_strategy == 'has_positive':
            label = num_pos_label > 0
        else:
            raise ValueError(args.oracle_experiment_label_strategy)

        final_predictions[example_idx] = label

    return {
        'predictions': final_predictions,
        'grouped_predictions': grouped_predictions,
        'model_path': args.model_name_or_path
    }


def run_paraphrase_identification_model(
    instances: List[Dict],
    model_path: Path,
    batch_size: int = 32,
    return_dict=True
):
    dummy_train_data = [
        {'idx': 'test-0', 'sent_1': 'hi', 'sent_2': 'hi', 'label': 0},
        {'idx': 'test-1', 'sent_1': 'hi', 'sent_2': 'hi', 'label': 0},
        {'idx': 'test-2', 'sent_1': 'hi', 'sent_2': 'hi', 'label': 1},
        {'idx': 'test-3', 'sent_1': 'hi', 'sent_2': 'hi', 'label': 1},
    ]

    with tempfile.NamedTemporaryFile() as tf:
        tmp_name = Path(tf.name).name
    tmp_name = f'to_prediction.{tmp_name}.csv'

    tmp_train_file = model_path / 'tmp_train.csv'
    dump_samples(dummy_train_data, tmp_train_file)
    pred_file_input = model_path / tmp_name
    dump_samples(instances, pred_file_input, is_test=False)

    train_args = [
        '--model_name_or_path', str(model_path),
        '--do_predict',
        # '--max_seq_length', 64,
        '--per_device_eval_batch_size', str(batch_size),
        '--output_dir', str(model_path),
        '--train_file', str(tmp_train_file),
        '--validation_file', str(tmp_train_file),
        '--test_file', str(pred_file_input),
        '--logging_dir', str(model_path),
        '--fp16'
    ]

    print(f'Running predictions on {pred_file_input} with {len(instances)} instances...')

    env = copy.copy(os.environ)
    env['WANDB_MODE'] = 'dryrun'

    python(
        '-m',
        'paraphrase.hf_paraphrase_identification',
        *train_args,
        _err=sys.stderr,
        _out=sys.stdout,
        _env=env
    )

    predictions = load_paraphrase_identification_model_prediction(
        pred_file_input, model_path / 'test_results_None.txt', return_dict)

    return predictions


if __name__ == '__main__':
    work_dir = Path('runs/iterative_learning_run_b92891/round1_iter1/')

    # main(
    #     work_dir / 'train.paraphrased.iter1.jsonl',
    #     output_prefix=work_dir / 'train.paraphrased.iter1.pi_sample1000',
    #     sample_size=1000
    # )

    # generate_training_data(
    #     work_dir / 'train.paraphrased.iter1.jsonl',
    #     dataset_file=work_dir / 'train.paraphrased.iter1.pi_sample1000.all.csv',
    #     prediction_file=work_dir / 'pi_model_sample1000_epoch1_lr2e-5/' / 'test_results_None.txt',
    #     output_file=work_dir / 'train.paraphrased.iter1.pi_model_filtered.jsonl'
    # )
    #
    # for sample_size in [3000]:
    #     for epoch in [2, 3, 5]:
    #         for ngsim in [False, True]:
    #             run(
    #                 SimpleNamespace(
    #                     work_dir=work_dir,
    #                     lr=2e-5,
    #                     epoch=epoch,
    #                     sample_size=sample_size,
    #                     sample_negative_example_by_sim_score=ngsim
    #                 )
    #             )

    for sample_size in [None]:
        for epoch in [2]:
            for ngsim in [True]:
                train_and_infer(
                    SimpleNamespace(
                        work_dir=work_dir,
                        lr=2e-5,
                        epoch=epoch,
                        sample_size=sample_size,
                        sample_negative_example_by_sim_score=ngsim,
                        model_name_or_path='/ocean/projects/dbs200003p/pcyin/Research/transformers/examples/text-classification/runs/qqp_lr2e-5_bs64_ms28425_warmup2800/checkpoint-11500'
                    )
                )

    train_and_infer(
        SimpleNamespace(
            work_dir=Path('runs/model-train-paraphrased-iter1-pi-none-lr2e-05-ep5-ngsimtrue-parser-input-jsonl-seed0-iter2/'),
            lr=2e-5,
            epoch=2,
            sample_size=None,
            sample_negative_example_by_sim_score=True,
            model_name_or_path=Path('runs/iterative_learning_run_b92891/round1_iter1/pi_None_lr2e-05_ep2_ngsimTrue_model/')
        )
    )
