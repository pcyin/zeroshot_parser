import ast
import json
import random
from types import SimpleNamespace
from typing import List, Dict, Mapping, Any
from pathlib import Path
import pyparsing
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from grammar_generation.program_utils import Sexp, normalize_program, parse_sexp_string, sexp_to_tokenized_string, \
    clone_sexp, replace_program, normalize_reverse_call

SCHOLAR_PLACEHOLDER_ENTITY_MAP = {
    'Dan Klein': {
        'type_name': 'authorname',
        'entity': 'fb:en.author.dan_klein',
    },
    'Tom Mitchell': {
        'type_name': 'authorname',
        'entity': 'fb:en.author.tom_mitchell'
    },
    'natural language processing': {
        'type_name': 'keyphrasename',
        'entity': 'fb:en.keyphrase.natural_language_processing'
    },
    'deep learning': {
        'type_name': 'keyphrasename',
        'entity': 'fb:en.keyphrase.deep_learning'
    },
    'machine learning': {
        'type_name': 'keyphrasename',
        'entity': 'fb:en.keyphrase.machine_learning'
    },
    '2012': {
        'type_name': 'year',
        'entity': ['number', '2012', 'year'],
    },
    '2016': {
        'type_name': 'year',
        'entity': ['number', '2016', 'year'],
    },
    '10': {
        'type_name': 'misc',
        'entity': ['number', '10', 'count']
    },
    'NIPS': {
        'type_name': 'venuename',
        'entity': 'fb:en.venue.nips'
    },
    'ICML': {
        'type_name': 'venuename',
        'entity': 'fb:en.venue.icml'
    },
    'Nature': {
        'type_name': 'journalname',
        'entity': 'fb:en.journal.nature'
    },
    'Wikipedia': {
        'type_name': 'datasetname',
        'entity': 'fb:en.dataset.wikipedia'
    },
    'ImageNet': {
        'type_name': 'datasetname',
        'entity': 'fb:en.dataset.imagenet'
    },
    'neural machine translation': {
        'type_name': 'title',
        'entity': 'fb:en.title.nmt'
    },
    'Multivariate Data Analysis': {
        'type_name': 'title',
        'entity': 'fb:en.title.multivariate_data_analysis'
    }
}

GEO_PLACEHOLDER_ENTITY_MAP = {
    'the US': {
        'type_name': 'country',
        'entity': 'fb:en.country.usa'
    },
    'sacramento': {
        'type_name': 'city',
        'entity': 'fb:en.city.sacramento_ca',
    },
    'austin': {
        'type_name': 'city',
        'entity': 'fb:en.city.austin_tx',
    },
    'california': {
        'type_name': 'state',
        'entity': 'fb:en.state.california',
    },
    'texas': {
        'type_name': 'state',
        'entity': 'fb:en.state.texas',
    },
    'colorado river': {
        'type_name': 'river',
        'entity': 'fb:en.river.colorado',
    },
    'red river': {
        'type_name': 'river',
        'entity': 'fb:en.river.red',
    },
    'lake tahoe': {
        'type_name': 'lake',
        'entity': 'fb:en.lake.tahoe',
    },
    'lake huron': {
        'type_name': 'lake',
        'entity': 'fb:en.lake.huron',
    },
    'mount whitney': {
        'type_name': 'mountain',
        'entity': 'fb:en.mountain.whitney',
    },
    'mount rainier': {
        'type_name': 'mountain',
        'entity': 'fb:en.mountain.rainier',
    },
    'death valley': {
        'type_name': 'place',
        'entity': 'fb:en.place.death_valley',
    },
    'pacific ocean': {
        'type_name': 'place',
        'entity': 'fb:en.place.pacific_ocean',
    },
    'san jose': {
        'type_name': 'name',
        'entity': 'fb:en.name.san_jose',
    },
}


PLACEHOLDER_ENTITY_MAP = {
    **SCHOLAR_PLACEHOLDER_ENTITY_MAP,
    **GEO_PLACEHOLDER_ENTITY_MAP
}


TYPE_NAMES = set(
    e['type_name']
    for e in PLACEHOLDER_ENTITY_MAP.values()
)


program_parser = pyparsing.nestedExpr(
    '(', ')',
    ignoreExpr=pyparsing.dblQuotedString.copy()
)


def load_samples(
    examples_path: str,
    score_path: str = None,
    num_samples: int = None,
    disabled_production_types: List[str] = None
) -> List[Dict]:
    examples = []
    disabled_production_types = set(disabled_production_types or [])
    num_skipped_examples = 0
    skipped_example_ids = set()
    for idx, line in tqdm(enumerate(
        Path(examples_path).open()),
        desc='Loading samples...',
        total=len(Path(examples_path).open().readlines())
    ):
        if num_samples and idx >= num_samples:
            break

        data = line.strip().split('\t')
        utterance = data[0]
        program = data[1]
        # program = program_parser.parseString(program.strip()).asList()
        program = parse_sexp_string(program.strip())
        program = normalize_program(program, normalize_filter_order=False)
        normalized_program = normalize_program(program)

        derivation = data[2]
        root_deriv_step = derivation.partition(') (')[0] + ')'
        root_deriv_step = ast.literal_eval(root_deriv_step)
        root_type = root_deriv_step[-1]

        # disabled_production_types
        if any(f"'{prod_type}'" in derivation for prod_type in disabled_production_types):
            skipped_example_ids.add(idx)
            num_skipped_examples += 1

        program_depth = int(data[3])

        post_processed_example = post_process_example(utterance, program, normalized_program)

        example = {
            'idx': idx,
            'program': program,
            'denotation_type': root_type,
            'complexity': program_depth,
            **post_processed_example
        }
        examples.append(example)

        # examples.append({
        #     'idx': idx,
        #     'utterance': utterance,
        #     'program': program,
        #     'denotation_type': root_type,
        #     'program_depth': program_depth
        #     #'derivation': derivation
        # })

    if score_path:
        print('loading scores')
        for line in Path(score_path).open():
            data = line.strip().split('\t')
            idx = int(data[0])
            score = float(data[1])

            if idx < len(examples):
                examples[idx]['score'] = score

    examples = [
        e
        for e in examples
        if e['idx'] not in skipped_example_ids
    ]

    # rank examples by scores
    if score_path:
        examples = sorted(
            examples,
            key=lambda e: -e['score']
        )

    print(f'Num skipped examples: {num_skipped_examples}')

    return examples


def group_samples(examples: List[Dict]) -> Mapping[str, List[Dict]]:
    # group examples
    examples_index = dict()
    for example in tqdm(examples, desc='Normalizing programs...'):
        # program = normalize_program(example['program'])
        # program_repr = ' '.join([str(x) for x in sexp_to_tokenized_string(program)])
        # example['canonical_program'] = program
        program_repr = example['canonical_lf']
        examples_index.setdefault(program_repr, []).append(example)

    has_score = examples[0].get('score') is not None
    if has_score:
        program_signatures = list(examples_index)
        for program_string in tqdm(program_signatures, desc='Ranking...'):
            examples = examples_index[program_string]
            ranked_examples = sorted(examples, key=lambda e: e['score'], reverse=True)
            examples_index[program_string] = ranked_examples

    return examples_index


def get_program_depth(program: Sexp) -> int:
    if isinstance(program, list):
        return 1 + max([get_program_depth(arg) for arg in program])
    else:
        return 0


def post_process_example(source: str, program: Sexp, normalized_program: Sexp = None):
    entities = []
    for entity, entry_map in PLACEHOLDER_ENTITY_MAP.items():
        if entity in source:
            entity_char_pos = source.index(entity)
            entities.append({
                'name': entity,
                'char_pos': entity_char_pos,
                **entry_map
            })

    # assign ID to placeholders
    for type_name in TYPE_NAMES:
        entities_with_type_name = sorted(
            [
                e
                for e in entities
                if e['type_name'] == type_name
            ],
            key=lambda e: e['char_pos']
        )

        for idx, entity in enumerate(entities_with_type_name):
            entity['indexed_type_name'] = f'{entity["type_name"]}{idx}'

    canonicalized_source = source
    canonicalized_lf = clone_sexp(program)
    if normalized_program:
        normalized_program = clone_sexp(normalized_program)
    variables = {}
    for entity in entities:
        canonicalized_source = canonicalized_source.replace(entity['name'], entity['indexed_type_name'])
        if entity['type_name'] in {'year', 'misc'}:
            canonicalized_lf = replace_program(canonicalized_lf, entity['entity'], [entity['indexed_type_name']])
            if normalized_program:
                normalized_program = replace_program(normalized_program, entity['entity'], [entity['indexed_type_name']])

            variables[entity['indexed_type_name']] = entity['name']
        else:
            canonicalized_lf = replace_program(canonicalized_lf, entity['entity'], entity['indexed_type_name'])
            if normalized_program:
                normalized_program = replace_program(normalized_program, entity['entity'], entity['indexed_type_name'])

            variables[entity['indexed_type_name']] = entity['entity']

    example = {
        'nl': canonicalized_source,
        'can': source,
        'lf': ' '.join(sexp_to_tokenized_string(canonicalized_lf)),
        'variables': variables
    }

    if normalized_program:
        example['canonical_lf'] = ' '.join(sexp_to_tokenized_string(normalized_program))

    return example


def generate_dataset_randomly(
    programs_by_complexity: Dict[int, List[List[Dict]]],
    num_examples: int,
    heldout_set: List[Dict] = None,
    seed=1234
):
    all_samples = []
    heldout_set = heldout_set or []
    heldout_examples_idx_set = {e['idx'] for e in heldout_set}
    for depth, examples_list in programs_by_complexity.items():
        for examples in examples_list:
            for e in examples:
                if e['idx'] not in heldout_examples_idx_set:
                    e['depth'] = depth
                    all_samples.append(e)

    dataset = []

    rng = np.random.RandomState(seed)
    rng.shuffle(all_samples)

    example_ptr = 0
    while example_ptr < len(all_samples) and len(dataset) < num_examples:
        example = all_samples[example_ptr]

        if is_valid_example(example):
            dataset.append({k: v for k, v in example.items() if k != 'program'})

        example_ptr += 1

    return dataset


def generate_dataset_by_lm_score(
    programs_by_complexity: Dict[int, List[List[Dict]]],
    top_k: int,
):
    all_samples = []
    for depth, examples_list in programs_by_complexity.items():
        for examples in examples_list:
            for e in examples:
                e['depth'] = depth

            all_samples.extend(examples)

    all_samples.sort(key=lambda e: e['score'], reverse=True)
    dataset = []

    example_ptr = 0
    while len(dataset) < top_k:
        example = all_samples[example_ptr]

        if is_valid_example(example):
            dataset.append({k: v for k, v in example.items() if k != 'program'})

        example_ptr += 1

    return dataset


def generate_dataset_by_program_depth(
    programs_by_complexity: Dict[int, List[List[Dict]]],
    lm_prob_threshold: float = 5.0,
    num_programs_each_depth: int = 1000,
    min_program_depth: int = 3,
    sample_strategy='rank_by_lm_score',
    seed: int = 1234
):
    assert sample_strategy in ['rank_by_lm_score', 'random']
    random.seed(seed)

    depths = [d for d in sorted(programs_by_complexity.keys()) if d >= min_program_depth]
    dataset = []

    num_examples_to_visualize = 100
    is_example_scored = next(iter(programs_by_complexity.values()))[0][0].get('score') is not None

    for depth in depths:
        examples_list = programs_by_complexity[depth]

        if is_example_scored and sample_strategy == 'rank_by_lm_score':
            print('Sort examples by LM score')
            examples_list = sorted(examples_list, key=lambda x: x[0]['score'], reverse=True)
        elif sample_strategy == 'random':
            print('Randomly shuffle examples...')
            random.shuffle(examples_list)
        else:
            raise ValueError(sample_strategy, is_example_scored)

        print(f'\n******* Depth: {depth} *******\n')
        valid_cnt = 0
        for e_id, examples in enumerate(examples_list):
            selected_examples = []
            best_example = examples[0]
            best_example['depth'] = depth

            if is_valid_example(
                best_example,
                selected_examples
            ):
                # processed_example = post_process_example(
                #     best_example['utterance'],
                #     best_example['program'],
                #     best_example['canonical_program']
                # )
                # processed_example.update({
                #     'idx': best_example['idx'],
                #     'score': best_example.get('score', 0.),
                #     'complexity': depth,
                #     'denotation_type': best_example['denotation_type']
                # })

                selected_examples.append({k: v for k, v in best_example.items() if k != 'program'})
                valid_cnt += 1

            for other_example in examples[1:]:
                other_score = other_example.get('score', 0.)
                best_score = best_example.get('score', 0.)
                if not is_example_scored or best_score - other_score <= lm_prob_threshold:
                    if is_valid_example(
                        other_example,
                        selected_examples
                    ):
                        # other_example['depth'] = depth
                        # processed_example = post_process_example(
                        #     other_example['utterance'],
                        #     other_example['program'],
                        #     other_example['canonical_program']
                        # )
                        # processed_example.update({
                        #     'idx': other_example['idx'],
                        #     'score': other_example.get('score', 0.),
                        #     'complexity': depth,
                        #     'denotation_type': other_example['denotation_type']
                        # })

                        selected_examples.append({k: v for k, v in other_example.items() if k != 'program'})

            dataset.extend(selected_examples)

            if e_id < num_examples_to_visualize and selected_examples:
                print(best_example['nl'])
                print(best_example['can'])
                print(best_example['lf'])
                if is_example_scored:
                    best_score = best_example['score']
                    print(best_score)
                print()
                if len(examples) > 1:
                    print('Alternatives:')
                    for example in examples[1:30]:
                        print(example['nl'])
                        print(example['can'])
                        if is_example_scored:
                            other_score = example['score']

                            score_diff = best_score - other_score
                            if score_diff <= lm_prob_threshold:
                                print(example['score'], '(Included)')
                            else:
                                print(example['score'])

                print('-' * 10)
                print()

            if valid_cnt >= num_programs_each_depth:
                break

    return dataset


def is_valid_example(
    example,
    selected_examples: List[Dict] = None
) -> bool:
    utterance = example['can']

    utterance_segments = utterance.split(' and ')
    keywords = [
        'keyword', 'author', 'journal', 'paper', 'topic', 'dataset', 'venue', 'title', 'keyphrase',  # SCHOLAR
        'country', 'city', 'state', 'river', 'lake', 'mountain', 'place', 'name',  # GEO
    ]
    prop = ['for', 'about', 'of', 'by', 'cites', 'in', 'titled', 'named']
    if utterance in keywords:
        return False

    flag = True
    for segment in utterance_segments:
        segment_tokens = segment.strip().split(' ')
        # of author, by paper, about keyword
        if segment_tokens[-1] in keywords and len(segment_tokens) > 1 and segment_tokens[-2] in prop and not ('number of' in utterance and len(utterance_segments) == 1):
            flag = False
            break
        elif segment in ['that paper cites']:
            flag = False
            break

    if not flag:
        return False

    for keyword in keywords:
        if (
            f"( string = ) ( call SW.getProperty ( call SW.singleton fb:en.{keyword} ) ( string ! type ) )" in example['lf'] or
            f"( string ! = ) ( call SW.getProperty ( call SW.singleton fb:en.{keyword} ) ( string ! type ) )" in example['lf']
        ):
            flag = False
            break

    for variable in example['variables']:
        if example['nl'].count(variable) > 1:
            flag = False
            break

    if not flag:
        return False

    if selected_examples and any(example['nl'] == e['nl'] for e in selected_examples):
        flag = False

    return flag


def main():
    arg_parser = ArgumentParser('dump dataset')
    arg_parser.add_argument('--strategy', type=str, choices=['by_program_depth', 'lm_score_desc', 'random'], default='by_program_depth')
    arg_parser.add_argument('--per-program-depth-sample-strategy', type=str, choices=['rank_by_lm_score', 'random'], default='rank_by_lm_score')
    arg_parser.add_argument('--lm-prob-threshold', type=float, default=7.0)
    arg_parser.add_argument('--num-programs-each-depth', type=int, default=1000)
    arg_parser.add_argument('--split-method', type=str, choices=['iid', 'template'], default='iid')
    arg_parser.add_argument('--samples-file', type=Path, required=True)
    arg_parser.add_argument('--num-samples', type=int, required=False, default=None)
    arg_parser.add_argument('--lm-scores-file', type=Path, default=None, required=False)
    arg_parser.add_argument('--output-file', type=Path, required=True)
    arg_parser.add_argument('--dev-ratio', type=float, default=0.1)
    arg_parser.add_argument('--disabled-production-types', nargs='+', type=str)

    args = arg_parser.parse_args()
    generate(args)


samples = None
samples_index_by_program_signature = None


def generate(args: SimpleNamespace, dev_set: List[Dict] = None, seed: int = 1234):
    global samples, samples_index_by_program_signature
    if samples is None or getattr(args, 'cache', False) is False:
        samples = load_samples(args.samples_file, args.lm_scores_file, args.num_samples, args.disabled_production_types)
        samples_index_by_program_signature = group_samples(samples)

    samples_by_complexity = dict()
    for program_string, examples in tqdm(
        samples_index_by_program_signature.items(),
        desc='Organizing by program complexity...'
    ):
        # program = examples[0]['program']
        # program_depth = get_program_depth(program)
        program_depth = examples[0]['complexity']
        samples_by_complexity.setdefault(program_depth, []).append(examples)

    if args.strategy == 'by_program_depth':
        dataset = generate_dataset_by_program_depth(
            samples_by_complexity,
            lm_prob_threshold=args.lm_prob_threshold,
            num_programs_each_depth=args.num_programs_each_depth,
            sample_strategy=args.per_program_depth_sample_strategy,
            seed=seed
        )
    elif args.strategy == 'lm_score_desc':
        dataset = generate_dataset_by_lm_score(
            samples_by_complexity,
            top_k=args.num_examples
        )
    elif args.strategy == 'random':
        dataset = generate_dataset_randomly(
            samples_by_complexity,
            num_examples=args.num_examples,
            # heldout_set=dev_set
        )
    else:
        raise ValueError(args.strategy)

    return output_split_dataset(
        dataset, args.dev_ratio,
        output_file=args.output_file, split_method=args.split_method,
        dev_set=dev_set
    )


def output_split_dataset(
    dataset: List[Dict],
    dev_ratio: 0.1,
    output_file: Path,
    split_method: str = 'iid',
    dev_set: List[Dict] = None
):
    np.random.seed(1234)

    if not dev_set:
        if split_method == 'iid':
            idx_list = list(range(len(dataset)))
            np.random.shuffle(idx_list)
            num_train_examples = int((1.0 - dev_ratio) * len(idx_list))

            train_examples_idx_list = [idx for idx in idx_list[:num_train_examples]]
            dev_examples_idx_list = [idx for idx in idx_list[num_train_examples:]]

            train_examples = [dataset[idx] for idx in train_examples_idx_list]
            dev_examples = [dataset[idx] for idx in dev_examples_idx_list]
        elif split_method == 'template':
            dataset_indexed_by_program_template = {}
            for example in dataset:
                dataset_indexed_by_program_template.setdefault(example['canonical_lf'], []).append(example)

            print(f'{len(dataset_indexed_by_program_template)} unique program templates in total...')
            program_templates = list(sorted(dataset_indexed_by_program_template))

            idx_list = list(range(len(program_templates)))
            np.random.shuffle(idx_list)

            num_train_templates = int((1 - dev_ratio) * len(idx_list))
            train_templates_idx_list = idx_list[:num_train_templates]
            dev_templates_idx_list = idx_list[num_train_templates:]

            train_templates = [program_templates[idx] for idx in train_templates_idx_list]
            dev_templates = [program_templates[idx] for idx in dev_templates_idx_list]

            train_examples = []
            for template in train_templates:
                train_examples.extend(dataset_indexed_by_program_template[template])

            dev_examples = []
            for template in dev_templates:
                dev_examples.extend(dataset_indexed_by_program_template[template])
        elif split_method == 'template_nat_sample':
            from scipy.special import logsumexp

            dataset_indexed_by_program_template = {}
            for example in dataset:
                dataset_indexed_by_program_template.setdefault(example['canonical_lf'], []).append(example)

            print(f'{len(dataset_indexed_by_program_template)} unique program templates in total...')
            program_templates = list(sorted(dataset_indexed_by_program_template))

            # compute the marginal naturalness of each template
            template_p_naturalness = []
            # Python 3.7 dict is ordered by default
            for template in program_templates:
                examples = dataset_indexed_by_program_template[template]

                p_nat = float(np.exp(logsumexp(np.array([e['score'] for e in examples]))))
                template_p_naturalness.append(p_nat)

            idx_list = list(range(len(program_templates)))
            template_p_naturalness = np.array(template_p_naturalness)
            template_p_naturalness = template_p_naturalness / template_p_naturalness.sum()

            print('Top K most likely templates:')
            print(template_p_naturalness[:100])

            num_dev_templates = int(dev_ratio * len(idx_list))
            dev_templates_idx_list = np.random.choice(idx_list, replace=False, p=template_p_naturalness, size=num_dev_templates)
            train_templates_idx_list = [idx for idx in idx_list if idx not in set(dev_templates_idx_list)]

            train_templates = [program_templates[idx] for idx in train_templates_idx_list]
            dev_templates = [program_templates[idx] for idx in dev_templates_idx_list]

            train_examples = []
            for template in train_templates:
                train_examples.extend(dataset_indexed_by_program_template[template])

            dev_examples = []
            for template in dev_templates:
                dev_examples.extend(dataset_indexed_by_program_template[template])
        else:
            raise ValueError(split_method)
    else:
        print('We use fixed dev_set')
        dev_example_idx_set = {e['idx'] for e in dev_set}
        dev_examples = dev_set
        train_examples = [e for e in dataset if e['idx'] not in dev_example_idx_set]

    train_file = output_file.with_suffix('.train.jsonl')
    with train_file.open('w') as f:
        for entry in train_examples:
            f.write(json.dumps(entry) + '\n')

    dev_file = output_file.with_suffix('.dev.jsonl')
    with dev_file.open('w') as f:
        for entry in dev_examples:
            f.write(json.dumps(entry) + '\n')

    with output_file.open('w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    return {
        'train_examples': train_examples,
        'dev_examples': dev_examples
    }


def test():
    # print(is_valid_example("paper"))
    # print(is_valid_example("paper by author"))
    # print(is_valid_example("keyword used by author and published in NIPS"))
    # print(is_valid_example("venue in keyword used by authorname0"))
    program = '( call SW.listValue ( call SW.filter ( call SW.filter ( call SW.getProperty ( call SW.singleton fb:en.paper ) ( string ! type ) ) ( string keyphrase_paper_keyphrase ) ( string = ) fb:en.keyphrase.deep_learning ) ( string citation_count_paper_number ) ( string < ) ( number 10 count ) ) )'
    program = parse_sexp_string(program)
    noramlized_program = normalize_program(program)
    example = post_process_example(
        'paper in deep learning and whose citation count is smaller than 10',
        program,
        noramlized_program
    )

    print(example)


if __name__ == '__main__':
    main()
    # test()
