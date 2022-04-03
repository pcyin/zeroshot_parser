"""
Fine tune the paraphraser using parser-filtered results
"""
import numpy as np
import json
import sys, os
from collections import OrderedDict
from pathlib import Path
import uuid
import logging
from typing import Dict, List

os.environ['MKL_THREADING_LAYER'] = 'GNU'


logger = logging.getLogger("finetune_paraphraser")
logger.setLevel(logging.INFO)


def get_valid_examples_from_index(canonical_example_idx: Dict, paraphrase_examples_index: Dict, example_ids: List):
    examples = []
    for idx in example_ids:
        paraphrase_examples = paraphrase_examples_index[idx]
        canonical_example = canonical_example_idx[idx]

        canonical_utterance = canonical_example['can']
        for paraphrase_example in paraphrase_examples:
            example = {
                'src': canonical_utterance,
                'tgt': paraphrase_example["can"]
            }

            examples.append(example)

    return examples


def main(dataset_file: Path, canonical_dataset_file: Path, init_parapharser_model: Path):
    suffix = str(uuid.uuid4())[:6]
    tmp_data_folder = dataset_file.parent / (Path(dataset_file.name).with_suffix(f'.tmp_{suffix}'))

    print(f'Create tmp data folder at {tmp_data_folder}')
    tmp_data_folder.mkdir(parents=True, exist_ok=True)

    canonical_examples = [
        json.loads(line)
        for line
        in canonical_dataset_file.open()
    ]

    canonical_examples_index = {
        str(example['idx']): example
        for example
        in canonical_examples
    }

    paraphrase_examples = [
        json.loads(line)
        for line
        in dataset_file.open()
    ]
    paraphrase_examples_index = OrderedDict()
    for example in paraphrase_examples:
        assert example.get('is_paraphrase')
        canonical_example_idx = example['idx'].partition('-')[0]
        if example['is_accepted_by_parser']:
            paraphrase_examples_index.setdefault(canonical_example_idx, []).append(example)

    np.random.seed(1234)
    all_example_ids = list(paraphrase_examples_index)
    np.random.shuffle(all_example_ids)
    num_train_example_ids = int(len(all_example_ids) * 0.8)

    train_example_ids = all_example_ids[:num_train_example_ids]
    dev_example_ids = all_example_ids[num_train_example_ids:]

    train_examples = get_valid_examples_from_index(canonical_examples_index, paraphrase_examples_index, train_example_ids)
    dev_examples = get_valid_examples_from_index(canonical_examples_index, paraphrase_examples_index, dev_example_ids)

    with (tmp_data_folder / 'train.source').open('w') as f_src, (tmp_data_folder / 'train.target').open('w') as f_tgt:
        for example in train_examples:
            f_src.write(f'{example["src"]}\n')
            f_tgt.write(f'{example["tgt"]}\n')

    with (tmp_data_folder / 'valid.source').open('w') as f_src, (tmp_data_folder / 'valid.target').open('w') as f_tgt:
        for example in dev_examples:
            f_src.write(f'{example["src"]}\n')
            f_tgt.write(f'{example["tgt"]}\n')

    wget('-N', 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json', '-P', tmp_data_folder)
    wget('-N', 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe', '-P', tmp_data_folder)
    wget('-N', 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt', '-P', tmp_data_folder)

    fairseq_root = Path('/projects/tir1/users/pengchey/Research/my_fairseq')

    for split in ['train', 'valid']:
        for lang in ['source', 'target']:
            python(
                fairseq_root / 'examples/roberta/multiprocessing_bpe_encoder.py',
                '--encoder-json', tmp_data_folder / 'encoder.json',
                '--vocab-bpe', tmp_data_folder / 'vocab.bpe',
                '--inputs', tmp_data_folder / f'{split}.{lang}',
                '--outputs', tmp_data_folder / f'{split}.bpe.{lang}',
                '--workers', '1',
                '--keep-empty',
                _out=sys.stdout,
                _err=sys.stderr
            )

    python(
        fairseq_root / 'fairseq_cli/preprocess.py',
        '--source-lang', 'source',
        '--target-lang', 'target',
        '--trainpref', tmp_data_folder / 'train.bpe',
        '--validpref', tmp_data_folder / 'valid.bpe',
        '--destdir', tmp_data_folder,
        '--workers', 1,
        '--srcdict', tmp_data_folder / 'dict.txt',
        '--tgtdict', tmp_data_folder / 'dict.txt',
        _out=sys.stdout,
        _err=sys.stderr
    )

    print(f'Generated {len(train_examples)} train examples, {len(dev_examples)} at {tmp_data_folder}')

    max_epoch = 5
    num_train_examples = len(train_examples)
    num_examples_per_update = 64
    batch_size = 64
    update_freq = num_examples_per_update // batch_size
    total_num_update = num_train_examples * max_epoch // num_examples_per_update
    warm_up_update = 500  # int(total_num_update * 0.2)

    print(dict(total_num_update=total_num_update, warm_up_update=warm_up_update, update_freq=update_freq))

    python(
        fairseq_root / 'fairseq_cli' / 'train.py',
        tmp_data_folder,
        '--restore-file', init_parapharser_model,
        '--batch-size', batch_size,
        '--task', 'translation',
        '--source-lang', 'source', '--target-lang', 'target',
        '--truncate-source',
        '--layernorm-embedding',
        '--share-all-embeddings',
        '--share-decoder-input-output-embed',
        '--reset-optimizer', '--reset-dataloader', '--reset-meters',
        '--required-batch-size-multiple', 1,
        '--arch', 'bart_large',
        '--criterion', 'label_smoothed_cross_entropy',
        '--label-smoothing', 0.1,
        '--dropout', 0.1, '--attention-dropout', 0.1,
        '--weight-decay', 0.01, '--optimizer', 'adam', '--adam-betas', '(0.9, 0.999)', '--adam-eps', '1e-08',
        '--clip-norm', 0.1,
        '--lr-scheduler', 'polynomial_decay',
        '--lr', 3e-5,
        '--total-num-update', total_num_update,
        '--max-update', total_num_update,
        '--warmup-updates', warm_up_update,
        '--update-freq', update_freq,
        '--fp16',
        '--save-dir', tmp_data_folder / 'checkpoints',
        '--best-checkpoint-metric', 'ppl',
        '--no-epoch-checkpoints',
        '--find-unused-parameters',
        _out=sys.stdout,
        _err=sys.stderr
    )


if __name__ == '__main__':
    from sh import wget, python

    main(
        dataset_file=Path('grammar_generation/data/all_derives_scholar_6.c07001cc.train.template.k1000.template_split.train.augmented.bs10.cdFalse.addqTrue.e467eb.parser_filtered.round2.addqfalse.jsonl'),
        canonical_dataset_file=Path('grammar_generation/data/all_derives_scholar_6.c07001cc.train.template.k1000.template_split.train.jsonl'),
        init_parapharser_model=Path(
            'data/paraphrase/'
            'checkpoints-maxtok512-updatefreq2-iter5000-warmup500/checkpoint_best.pt'
        )
    )
