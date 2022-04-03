"""Entry point for running experiments."""
import copy
import os
import uuid
from types import SimpleNamespace
from typing import Text, List, Dict, Any
import pathlib


Path = pathlib.Path


COMMAND_LINE_HEADER = ''

# Overload the following parameters.
EVALUATION_SERVER_ADDRESS = dict(
    scholar='http://vivace.clab.cs.cmu.edu:8081/',
    geo='http://vivace.clab.cs.cmu.edu:8082/'
)

IS_DRY_RUN = (os.getenv('DRY_RUN', default=None) is not None)


CONFIG_GROUPS = {
    'scholar': {
        'vanilla_canonical_lf': {
            '--seed-train-file': 'grammar_generation/data/canonical_examples.scholar.K_2000.jsonl',
            '--eval-file': 'data/train_dev_scholar_nat.cleaned.entity_fixed.migrated.json',
            '--test-file': 'data/test_scholar.json',
            '--end-iteration': 3,
            '--paraphraser-model-file': 'data/paraphrase/checkpoints-maxtok512-updatefreq2-iter5000-warmup500/checkpoint_best.pt',
            '--paraphraser-include-question': None,
            '--paraphraser-heuristic-deduplicate': None,
            '--paraphraser-seed-file-type': 'filtered',
            '--parser-validation-metric': 'ppl',
            '--parser-use-cumulated-datasets': None,
            '--parser-not-use-canonical-example': None,
            '--parser-logical-form-data-field': 'canonical_lf',
            # '--parser-max-epoch': 2  # The following toyish parameters are for debugging.
            # '--paraphraser-beam-size': 2
        },
    }
}


CONFIG_GROUPS.update({
    'geo': {
        'vanilla_canonical_lf': {
            **CONFIG_GROUPS['scholar']['vanilla_canonical_lf'],
            '--seed-train-file': 'grammar_generation/data/canonical_examples.geo.K_2000.jsonl',
            '--eval-file': 'data/train_dev_geo_nat.processed.migrated.json',
            '--test-file': 'data/test_geo.processed.json',
        },
    }
})


TRAIN_FILE_NAMES = {
    'scholar': {
        'by_k': [
            f'canonical_examples.scholar.K_{k}.jsonl'
            for k
            in [500, 1000, 2000, 4000, 8000]
        ],
    },
    'geo': {
        'by_k': [
            f'canonical_examples.geo.K_{k}.jsonl'
            for k
            in [500, 1000, 2000, 4000, 8000]
        ]
    }
}


def run(cmd: str, *args, work_dir: Path = None, run_file_name: Text = 'run.sh'):
    """Run a command line."""
    work_dir = work_dir or Path('.')

    commands = []
    for arg in args:
        commands.append(str(arg))

    command_line = ' '.join(commands)

    print(f'Run command {cmd} {command_line}')

    script = f"""#!/bin/bash

{cmd} {command_line}"""

    if not IS_DRY_RUN:
        (work_dir / run_file_name).open('w').write(script)
        os.system(f'bash {work_dir / run_file_name}')

    return SimpleNamespace(work_dir=work_dir)


def submit(config: Dict[Text, Any], run_prefix: Text, domain: Text):
    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    if not IS_DRY_RUN:
        train_file = config['--seed-train-file']
        if not Path(train_file).exists():
            raise ValueError(f'Training file {train_file} does not exist.')

        if '--seed-dev-file' in config:
            dev_file = config['--seed-dev-file']
            if not dev_file.exists():
                raise ValueError(f'Dev file {train_file} does not exist.')

        if '--eval-file' in config:
            assert Path(config['--eval-file']).exists()

    suffix = str(uuid.uuid4())[:6]
    work_dir = Path(f'runs/iterative_learning_{run_prefix}_{suffix}')

    config['--work-dir'] = work_dir

    assert not work_dir.exists()
    work_dir.mkdir(parents=True)

    arg_list = list()
    for key, val in config.items():
        arg_list.append(key)
        if val is not None:
            if isinstance(val, (tuple, list)):
                val = ' '.join(str(x) for x in val)
            else:
                val = str(val)

            arg_list.append(val)

    eval_server_addr = EVALUATION_SERVER_ADDRESS[domain]

    run_inst = run(
        (
            (f'{COMMAND_LINE_HEADER} && ' if COMMAND_LINE_HEADER else '') +
            f'export EVALUATION_SERVER_ADDR="{eval_server_addr}" && ' +
            f'PYTHONPATH=. python exp/multi_round_paraphrasing_and_parsing.py'
        ),
        *arg_list,
        work_dir=work_dir,
    )

    return run_inst


def sweep_seed(
    config_dict: Dict[Text, Any],
    domain: Text,
    seeds: List[int],
    run_prefix: Text,
):
    """Run experiments given the command arguments and a seed."""
    run_infos = []
    for seed in seeds:
        config_dict.update({
            '--seed': seed,
        })

        run = submit(
            config_dict,
            run_prefix=run_prefix,
            domain=domain
        )

        run_infos.append(run)

    return run_infos


def run_experiment_using_generated_dev_set(
    domain: Text,
    run_group='vanilla_canonical_lf',
    seeds: List[int] = None,
    train_file_name: Text = None,
    tag: Text = 'default'
):
    seeds = seeds or [0, 1, 2, 3, 4]
    data_root = Path('grammar_generation/data/')

    config_dict = copy.deepcopy(CONFIG_GROUPS[domain][run_group])

    if train_file_name:
        config_dict['--seed-train-file'] = str(data_root / train_file_name)

    for seed in seeds:
        _config_dict = copy.deepcopy(config_dict)
        _config_dict.update({
            '--no-paraphrase-dev-set': None,
        })

        run_insts = sweep_seed(
            _config_dict,
            domain=domain,
            seeds=[seed],
            run_prefix=tag
        )

        assert len(run_insts) == 1
        run_inst = run_insts[0]
        print(f'*** Initial Model Run: {run_inst.work_dir}')

        prev_run_dir = run_inst.work_dir

        init_model_work_dir = prev_run_dir / 'round1_iter2'

        dev_file = init_model_work_dir / 'dev.nat_iid_sample.k2000.jsonl'

        run(
            f'export WORK_DIR="{init_model_work_dir}" && '
            f'bash grammar_generation/generate_dev_data.sh',
            work_dir=init_model_work_dir,
            run_file_name='run.generate.dev_data.sh',
        )

        main_run_config = copy.deepcopy(config_dict)
        main_run_config.update({
            '--seed-dev-file': dev_file,
            '--no-paraphrase-dev-set': None,
            '--start-iteration': 1,
            '--end-iteration': 3,
            '--seed-model-file': dev_file.with_name('model.tar.gz'),
            '--parser-validation-after-epoch': 10,
        })

        sweep_seed(
            main_run_config,
            domain=domain,
            seeds=[seed * 10],
            run_prefix=tag
        )


def main():
    run_experiment_using_generated_dev_set('scholar', seeds=[0])
    run_experiment_using_generated_dev_set('geo', seeds=[0])


if __name__ == '__main__':
    main()
