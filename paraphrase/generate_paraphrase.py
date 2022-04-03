import gc
import itertools
import json
import math
import string
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import spacy
import torch
from allennlp.predictors import Predictor
from spacy.lang.en import English
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Iterator, Any, Optional, Union, Callable

from grammar_generation.lm_scoring import PretrainedLM
from paraphrase.paraphrase_tree import ParaphraseTree
from paraphrase.paraphraser import BartParaphraser, Paraphraser
from grammar_generation.generate_dataset_from_sampled_examples import PLACEHOLDER_ENTITY_MAP

from allennlp.models import load_archive

from grammar_generation.program_utils import normalize_program, parse_sexp_string
from paraphrase.sim.sim_utils import load_sim_model
from paraphrase.sim.test_sim import find_similarity
from paraphrase.utils import get_batches, normalize_utterance

WH_QUESTION_PREFIXES = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how']
WH_QUESTION_PREFIXES = WH_QUESTION_PREFIXES + [tok.capitalize() for tok in WH_QUESTION_PREFIXES]


def max_by(sequence: List[Any], key_func: Callable) -> Any:
    best_item = None
    best_score = -float('inf')
    for item in sequence:
        if key_func(item) > best_score:
            best_item = item
            best_score = key_func(item)

    return best_item


def generate_example_from_paraphrase_hypothesis(
    hyp: Dict,
    example: Dict,
    variables: Dict,
    idx: str
):
    sentence = hyp['sentence']
    canonical_sentence = str(sentence)
    for variable in variables.values():
        assert variable['mention'].lower() in sentence

        canonical_sentence = canonical_sentence.replace(variable['mention'].lower(), variable['canonical_name'])

    hyp_example = {
        'idx': f'{example["idx"]}-paraphrase-{idx}',
        'nl': canonical_sentence,
        'source_nl': example['nl'],
        'original_canonical_nl': (
            example['original_canonical_nl']
            if example.get('is_paraphrase')
            else example['can']
        ),
        'can': sentence,
        'is_paraphrase': True,
        'score': hyp['score'] + example.get('score', 0.),
        'paraphrase_score': hyp['score'],
        'is_question_paraphrase': hyp.get('is_question_paraphrase', False),
        'paraphraser_idx': hyp.get('paraphraser_idx')
    }

    if 'paraphrase' not in str(example['idx']):
        hyp_example['original_lm_score'] = example.get('score', 0.)

    for key, val in example.items():
        if key not in {
            'sim_score', 'score', 'idx', 'is_paraphrase',
            'is_valid_paraphrase',
            'is_accepted_by_parser', 'accepted_by_parser_in_iter',
            'paraphrase_identification_labels',
            'pruner_metadata',
            'pid_model_metadata'
        } and key not in hyp_example:
            hyp_example[key] = val

    # for key in ['complexity', 'variables', 'lf']:
    #     if key in example:
    #         hyp_example[key] = example[key]

    return hyp_example


def gather_example_variable_info(
    example: Dict
) -> Dict:
    """
    {
        'authorname0': {}
    }
    """
    variable_info = {}
    for placeholder_name, entity_value in example['variables'].items():
        entity_mention = [
           m
           for m, v in
           PLACEHOLDER_ENTITY_MAP.items()
           if (
                v['entity'] == entity_value or
                v['type_name'] in {'year', 'misc'} and entity_value == v['entity'][1]
            )
        ][0]

        variable_info[placeholder_name] = {
            'mention': entity_mention,
            'value': entity_value,
            'canonical_name': placeholder_name
        }

    return variable_info


def clean_hyp_paraphrase(sentence: str):
    for punct in string.punctuation:
        sentence = sentence.strip(punct)

    return sentence


def get_batch(
    dataset: List[Any],
    batch_size: int
) -> Iterator[List[Any]]:
    batch_num = math.ceil(len(dataset) / batch_size)
    for i in range(batch_num):
        examples = dataset[i * batch_size: (i + 1) * batch_size]

        yield examples


def is_valid_hyp(hyp_sentence: str, example: Dict, existing_examples: List[Dict]) -> bool:
    normalized_sent = hyp_sentence.encode('ascii', 'ignore').decode()

    if '  ' in normalized_sent:
        return False

    if len(normalized_sent.strip()) == 0:
        return False

    variable_info = example['variable_info']

    is_variables_intact = all(
        x['mention'].lower() in hyp_sentence
        for x
        in variable_info.values()
    )

    if not is_variables_intact:
        return False

    hyp_already_in_result_set = (
        any(
            hyp_['can'].lower() == hyp_sentence.lower()
            for hyp_
            in existing_examples
        )
    )

    if hyp_already_in_result_set:
        return False

    # check if there is repetitive mentions of variables
    for placeholder_name, entry in variable_info.items():
        mention = entry['mention'].lower()
        if hyp_sentence.count(mention) > 1:
            return False

    return True


def get_question_prefix_tokens_for_sentence(sentence: str, example: Dict) -> Optional[List[str]]:
    if '__sexp' not in example:
        example['__sexp'] = parse_sexp_string(example['lf'])

    program_sexp = example['__sexp']

    prefix_tokens = None

    # if sentence.startswith('author') or sentence.startswith('most cited author'):
    #     prefix_tokens = 'who'
    # elif sentence.startswith('paper') or sentence.startswith('most cited paper') or sentence.startswith('venue') or sentence.startswith('journal'):
    #     prefix_tokens = 'what'
    # elif sentence.startswith('number of') or sentence.startswith('citation'):
    #     prefix_tokens = 'how many'

    denotation_type = example['denotation_type']
    if denotation_type == '_author_':
        prefix_tokens = ['Who']
    elif denotation_type == '_number_':
        last_op = program_sexp[2]  # ( call SW.listValue ( call getProperty ( ... ) ( string population_state_count ) ) )
        prefix_tokens = ['What']
        if last_op[1] == 'SW.getProperty':
            relation = last_op[3][1]
            assert isinstance(relation, str)
            if 'publication_year' in relation:
                prefix_tokens = ['When', 'Which year']
            elif 'length' in relation:
                prefix_tokens = ['How long', 'What']
            elif 'citation' in relation:
                prefix_tokens = ['How many']
            elif 'population_city_count' in relation or 'population_state_count' in relation:
                prefix_tokens.append('How many')
        elif last_op[1] == '.size':
            prefix_tokens = ['How many']
    elif denotation_type == '_year_':
        prefix_tokens = ['When', 'Which year']
    elif denotation_type == '_venue_':
        prefix_tokens = ['Where']
    elif denotation_type in ['_state_', '_city_', '_place_']:
        prefix_tokens = ['What', 'Where']
    else:
        prefix_tokens = ['What']

    return prefix_tokens


def get_question_prefix_tokens_for_sentences(sentences: List[str]) -> List[Optional[List[str]]]:
    return [
        get_question_prefix_tokens_for_sentence(sent)
        for sent
        in sentences
    ]


def get_sim_score(
    batched_paraphrase_examples: List[List[Dict]],
    batched_examples: List[Dict],
    model: Dict,
    batch_size: int
) -> List[float]:
    dataset = []
    for example, hyps in zip(batched_examples, batched_paraphrase_examples):
        for hyp in hyps:
            dataset.append((hyp['can'], example['can']))

    all_scores = []
    for batch in (
        get_batches(dataset, batch_size=batch_size)
    ):
        # print(batch[0])
        sim_scores = find_similarity(
            [x[0] for x in batch],
            [x[1] for x in batch],
            model['tokenizer'],
            model['spm'],
            model['model']
        )

        all_scores.extend(sim_scores)

    ptr = 0
    for example, hyps in zip(batched_examples, batched_paraphrase_examples):
        for hyp in hyps:
            hyp['sim_score'] = all_scores[ptr]
            ptr += 1

    return all_scores


@torch.no_grad()
def is_valid_paraphrased_example_according_to_parser_model(
    parser_predictor: Predictor,
    batched_paraphrased_examples: Union[List[List[Dict]], List[Dict]],
    batch_size: int,
    batched_examples: List[Dict] = None,
    beam_size: int = 5,
    allowed_rank: int = 1
) -> List[Dict]:
    # print(f'number examples for prediction: {len(paraphrased_examples)}')
    assert allowed_rank <= beam_size

    instances = []
    if isinstance(batched_paraphrased_examples[0], list):
        batched_paraphrased_examples = chain(*batched_paraphrased_examples)

    for paraphrase_example in batched_paraphrased_examples:
        example_instance = parser_predictor._dataset_reader.text_to_instance(
            source_sequence=paraphrase_example['nl'],
            target_sequence=paraphrase_example['lf']
        )

        instances.append((example_instance, paraphrase_example, paraphrase_example['lf']))

    for batch in get_batch(instances, batch_size):
        instances_batch, examples_batch, gold_lf_list = zip(*batch)
        instances_batch = list(instances_batch)
        pred_results = parser_predictor.predict_batch_instance(instances_batch)

        canonical_tgt_programs = [
            normalize_program(parse_sexp_string(lf))
            for lf
            in gold_lf_list
        ]

        for hyp_idx, hyp_decode_result in enumerate(pred_results):
            paraphrased_example = examples_batch[hyp_idx]
            canonical_tgt_program = canonical_tgt_programs[hyp_idx]

            canonical_hyp_programs = []
            is_match = False
            for i in range(allowed_rank):
                hyp_program_tokens = hyp_decode_result['predictions'][i]['tokens']
                try:
                    hyp_program = parse_sexp_string(' '.join(hyp_program_tokens))
                    canonical_hyp_program = normalize_program(hyp_program)
                except:
                    canonical_hyp_program = ['ERROR']

                canonical_hyp_programs.append({
                    'program': canonical_hyp_program,
                    'score': hyp_decode_result['predicted_log_probs'][0]
                })

                if canonical_hyp_program == canonical_tgt_program:
                    is_match = True
                    paraphrased_example['parser_score'] = hyp_decode_result['predicted_log_probs'][i]
                    paraphrased_example['parser_hyp_rank'] = i
                    break

            paraphrased_example['is_accepted_by_parser'] = is_match

    # return [
    #     {
    #         'is_accepted_by_parser': paraphrased_example['is_accepted_by_parser'],
    #         'parser_score': paraphrased_example['parser_score']
    #     }
    #     for paraphrased_example
    #     in paraphrased_examples
    # ]


def deduplicate_examples(paraphrased_examples: List[Dict], nlp_model: English, paraphrase_tree: ParaphraseTree = None):
    paraphrased_examples = sorted(paraphrased_examples, key=lambda hyp: hyp['score'], reverse=True)

    filtered_examples = []
    existing_normalized_utterances = set()

    if paraphrase_tree and paraphrased_examples:
        root_idx = paraphrase_tree.get_root_idx(paraphrased_examples[0]['idx'])
        descents: List[Dict] = paraphrase_tree.get_descent_examples(root_idx)
        existing_normalized_utterances.update(
            set(e['normalized_can'] for e in descents)
        )

    parsed_docs = list(nlp_model.pipe([e['can'] for e in paraphrased_examples]))
    for example, doc in zip(paraphrased_examples, parsed_docs):
        normalized_utterance = normalize_utterance(doc)
        if normalized_utterance not in existing_normalized_utterances:
            filtered_examples.append(example)
            existing_normalized_utterances.add(normalized_utterance)
        else:
            pass

    return filtered_examples


@torch.no_grad()
def main(
    input_dataset_file: Path,
    output_file: Path,
    model_path: List[Path],
    batch_size: int,
    beam_size: int,
    parser_batch_size: int = 64,
    lm_scorer: Optional[str] = None,
    constrained_decoding: bool = False,
    include_question: bool = False,
    include_statement: bool = False,
    include_source_examples: bool = True,
    parser_model_file: Optional[Path] = None,
    only_keep_parser_filtered_examples: bool = False,
    paraphraser_arg_string: str = None,
    do_not_purge_beam: bool = False,
    sim_model_path: Optional[Path] = None,
    heuristic_deduplicate: bool = False,
    parser_allowed_rank_in_beam: int = 1,
    paraphrase_tree: Optional[ParaphraseTree] = None
):
    # if constrained_decoding:
    #     assert batch_size == 1, 'constrained decoding does not support batching!'

    dataset: List[Dict] = []
    for line in input_dataset_file.open():
        example = json.loads(line)
        dataset.append(example)

    print('init paraphraser')
    paraphrasers = []
    for path in model_path:
        print(f'Loading paraphraser: {path}')
        assert path.exists()
        model = Paraphraser.from_pretrained(path)
        #model = BartParaphraser(path)
        paraphrasers.append(model)

    nlp_model = spacy.load("en_core_web_md")
    print('loaded spacy model')

    sim_model = None
    if sim_model_path:
        print(f'Loading SIM model from {sim_model_path}')
        sim_model = load_sim_model(str(sim_model_path), cuda=torch.cuda.is_available())

    default_model_kwargs = {}
    if isinstance(paraphrasers[0], BartParaphraser):
        default_model_kwargs['normalize_scores'] = False

    if paraphraser_arg_string:
        model_extra_config = json.loads(paraphraser_arg_string)
        default_model_kwargs.update(model_extra_config)
    print(f'default args for paraphraser: {default_model_kwargs}')

    parser_model = None
    parser_predictor = None
    use_parser_for_filtering = False
    parser_filtering_result_list = []
    if parser_model_file:
        use_parser_for_filtering = True
        print(f'load parser at {parser_model_file} for filtering generated paraphrases...')

        model_archive = load_archive(parser_model_file)
        if torch.cuda.is_available():
            model_archive.model.cuda()

        parser_predictor = Predictor.from_archive(model_archive)
        parser_predictor._model._validation_metric = None

    augmented_dataset = []
    # indexed by the id of its originating canonical example
    paraphrase_results = OrderedDict()

    for iter_idx, batched_examples in enumerate(tqdm(
        get_batch(dataset, batch_size=batch_size),
        desc='Paraphrasing...', total=math.ceil(len(dataset) / batch_size)
    ), start=1):
        if torch.cuda.is_available() and iter_idx % 30 == 0:
            print('Clear up memory.....')
            gc.collect()
            torch.cuda.empty_cache()

        src_sentences: List[str] = [
            e['can']
            for e
            in batched_examples
        ]

        model_kwargs = dict(default_model_kwargs)

        for example in batched_examples:
            variable_info = gather_example_variable_info(example)
            example['variable_info'] = variable_info

            if constrained_decoding:
                model_kwargs['constraints'] = 'unordered'
                constrained_tokens = []
                for placeholder_name, entry in variable_info.items():
                    constrained_tokens.append(entry['mention'])

                model_kwargs.setdefault(
                    'constraint_tokens', []
                ).append(constrained_tokens)

        if constrained_decoding:
            any_constrained_tokens = any(
                len(token_list) > 1
                for token_list
                in model_kwargs['constraint_tokens']
            )

            if not any_constrained_tokens:
                model_kwargs.pop('constraints')
                model_kwargs.pop('constraint_tokens')

        blocked_tokens = None
        if include_statement:
            assert include_question
            blocked_tokens = [WH_QUESTION_PREFIXES for _ in range(len(batched_examples))]

        batched_paraphrases = [list() for _ in range(len(src_sentences))]

        for model_idx, paraphraser in enumerate(paraphrasers):
            kwargs = dict(model_kwargs)
            if isinstance(paraphraser, BartParaphraser):
                kwargs['blocked_tokens'] = blocked_tokens

            batched_paraphrases_ = paraphraser.generate(
                src_sentences,
                beam_size=beam_size,
                **kwargs
            )

            for hyp_list in batched_paraphrases_:
                for hyp in hyp_list:
                    hyp['paraphraser_idx'] = model_idx

            if include_question:
                valid_sentences = []
                valid_sentences_ids = []
                valid_sentences_prefix_tokens = []

                for idx, (sent, example) in enumerate(zip(src_sentences, batched_examples)):
                    prefix_tokens = get_question_prefix_tokens_for_sentence(sent, example) or []

                    if isinstance(prefix_tokens, str):
                        prefix_tokens = [prefix_tokens]

                    for prefix_token in prefix_tokens:
                        valid_sentences_ids.append(idx)
                        valid_sentences.append(sent)
                        valid_sentences_prefix_tokens.append(prefix_token)

                if valid_sentences:
                    model_kwargs_ = dict(default_model_kwargs)
                    model_kwargs_['prefix_tokens'] = valid_sentences_prefix_tokens
                    batched_question_paraphrases = paraphraser.generate(
                        valid_sentences,
                        beam_size=beam_size,
                        **model_kwargs_
                    )

                    for rel_idx, abs_idx in enumerate(valid_sentences_ids):
                        hyps = batched_question_paraphrases[rel_idx]
                        for hyp in hyps:
                            hyp['is_question_paraphrase'] = True

                        batched_paraphrases_[abs_idx].extend(hyps)

            for example_idx, hyp_list in enumerate(batched_paraphrases_):
                batched_paraphrases[example_idx].extend(hyp_list)

        batched_paraphrase_examples = []
        for example_idx, (example, paraphrases) in enumerate(
            zip(batched_examples, batched_paraphrases)
        ):
            variable_info = example['variable_info']
            paraphrased_examples = []
            for hyp_idx, hyp in enumerate(paraphrases):
                hyp_sentence = hyp['sentence']
                hyp_sentence = hyp_sentence.lower().strip()
                hyp_sentence = clean_hyp_paraphrase(hyp_sentence)

                is_valid = is_valid_hyp(hyp_sentence, example, paraphrased_examples)

                # if we include source examples, then we remove any samples that are the same with the source
                if include_source_examples and hyp_sentence.lower() == example['can'].lower():
                    continue

                if is_valid:
                    hyp['sentence'] = hyp_sentence

                    paraphrase_example = generate_example_from_paraphrase_hypothesis(hyp, example, variable_info, idx=hyp_idx)
                    paraphrased_examples.append(paraphrase_example)

            batched_paraphrase_examples.append(paraphrased_examples)

        if sim_model:
            get_sim_score(batched_paraphrase_examples, batched_examples, sim_model, batch_size=512)

        if use_parser_for_filtering:
            # inplace label examples
            is_valid_paraphrased_example_according_to_parser_model(
                parser_predictor, batched_paraphrase_examples,
                allowed_rank=parser_allowed_rank_in_beam,
                batch_size=parser_batch_size
            )

            parser_filtering_result_list.append([
                e['is_accepted_by_parser']
                for batch_ in batched_paraphrase_examples
                for e in batch_
            ])

        for example_idx, (example, paraphrased_examples) in enumerate(
            zip(batched_examples, batched_paraphrase_examples)
        ):
            # keep the top beam-size examples
            paraphrased_examples = sorted(
                paraphrased_examples, key=lambda x: x['paraphrase_score'], reverse=True)  # [:beam_size]

            if only_keep_parser_filtered_examples:
                assert parser_model_file
                paraphrased_examples = [e for e in paraphrased_examples if e['is_accepted_by_parser']]

            if heuristic_deduplicate:
                paraphrased_examples = deduplicate_examples(paraphrased_examples, nlp_model)

            # if larger than beam size, we keep at least half questions?
            if include_question and len(paraphrased_examples) > beam_size and not do_not_purge_beam:
                num_questions_to_keep = beam_size // 2
                pruned_paraphrased_examples = (
                    [
                        e
                        for e
                        in paraphrased_examples if not e['is_question_paraphrase']
                    ][:beam_size - num_questions_to_keep]
                ) + (
                    [
                        e
                        for e
                        in paraphrased_examples if e['is_question_paraphrase']
                    ][:num_questions_to_keep]
                )
                paraphrased_examples = pruned_paraphrased_examples

            batched_paraphrase_examples.append(paraphrased_examples)

            # original_canonical_example_idx = str(example['idx']).partition('-')[0]
            original_canonical_example_idx = ParaphraseTree.get_root_idx(str(example['idx']))

            if include_source_examples:
                example['is_source_example'] = True
                paraphrase_results.setdefault(original_canonical_example_idx, []).append(example)

            for example_ in paraphrased_examples:
                paraphrase_results.setdefault(original_canonical_example_idx, []).append(example_)

    if paraphrase_tree:
        grouped_paraphrased_examples = OrderedDict()
        for canonical_example_idx, paraphrased_examples in paraphrase_results.items():
            root_idx = paraphrase_tree.get_root_idx(canonical_example_idx)
            grouped_paraphrased_examples.setdefault(root_idx, []).extend(paraphrased_examples)

        for root_idx, paraphrased_examples in grouped_paraphrased_examples.items():
            existing_normalized_utterances = set()

            descents: List[Dict] = paraphrase_tree.get_descent_examples(root_idx)
            existing_normalized_utterances.update(
                set(e['normalized_can'] for e in descents)
            )

            parsed_docs = list(nlp_model.pipe([e['can'] for e in paraphrased_examples]))
            for example, doc in zip(paraphrased_examples, parsed_docs):
                normalized_utterance = normalize_utterance(doc)
                example['normalized_can'] = normalized_utterance

            grouped_examples_ = itertools.groupby(
                sorted(paraphrased_examples, key=lambda e: e['normalized_can']),
                key=lambda e: e['normalized_can']
            )

            for normalized_utterance, examples in grouped_examples_:
                examples = list(examples)

                if normalized_utterance not in existing_normalized_utterances:
                    example_with_best_score = max_by(examples, lambda e: e['score'])
                    augmented_dataset.append(example_with_best_score)
    else:
        def _deduplicate(_examples: List[Dict]) -> List[Dict]:
            _cleaned_examples = []
            for e in _examples:
                if any(
                        e['nl'].lower() == e2['nl'].lower()
                        for e2
                        in _cleaned_examples
                ):
                    continue

                _cleaned_examples.append(e)

            if heuristic_deduplicate:
                _cleaned_examples = deduplicate_examples(_cleaned_examples, nlp_model, paraphrase_tree)

            return _cleaned_examples

        # deduplicate examples with the same paraphrase
        for canonical_example_idx in list(paraphrase_results.keys()):
            root_idx = paraphrase_tree
            examples = paraphrase_results[canonical_example_idx]
            examples.sort(
                key=lambda e: (0 if e.get('is_source_example') else 1, -e['score'])
            )

            examples = _deduplicate(examples)
            paraphrase_results[canonical_example_idx] = examples

        for idx, examples in paraphrase_results.items():
            augmented_dataset.extend(examples)

    with output_file.open('w') as f:
        for example in augmented_dataset:
            f.write(json.dumps(example) + '\n')

    result_dict = {}
    if use_parser_for_filtering:
        r = list(chain(*parser_filtering_result_list))
        avg_parser_acc = np.average(r)

        print(f'Average parser acceptance acc. ({len(r)} samples): {avg_parser_acc}')

        result_dict['parser_accept_rate'] = avg_parser_acc

    for paraphraser in paraphrasers:
        paraphraser.cpu()
        if parser_model_file:
            model_archive.model.cpu()
            del model_archive, parser_predictor

        if sim_model_path:
            del sim_model

        del paraphrasers

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if lm_scorer:
        lm = PretrainedLM(
            lm_scorer,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )

        for batched_examples in get_batch(augmented_dataset, batch_size=batch_size):
            lm_scores = lm.get_sentence_likelihood([e['can'] for e in batched_examples])
            for idx, example in enumerate(batched_examples):
                example['lm_score'] = lm_scores[idx]

        del lm

    return augmented_dataset, result_dict


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--input-dataset-file', type=Path, required=True)
    arg_parser.add_argument('--output-file', type=Path, required=True)
    arg_parser.add_argument('--model-path', type=Path, nargs='+', required=True)
    arg_parser.add_argument('--batch-size', type=int, default=10)
    arg_parser.add_argument('--beam-size', type=int, default=5)
    arg_parser.add_argument('--constrained-decoding', action='store_true')
    arg_parser.add_argument('--include-question', action='store_true')
    arg_parser.add_argument('--include-statement', action='store_true')
    arg_parser.add_argument('--parser-model-file', type=Path, default=None)
    arg_parser.add_argument('--only-keep-parser-filtered-examples', action='store_true')
    arg_parser.add_argument('--include-source-examples', action='store_true')
    arg_parser.add_argument('--paraphraser-arg-string', type=str)
    arg_parser.add_argument('--do-not-purge-beam', action='store_true')
    arg_parser.add_argument('--sim-model-path', type=Path)
    arg_parser.add_argument('--heuristic-deduplicate', action='store_true')
    arg_parser.add_argument('--parser-allowed-rank-in-beam', type=int, default=1)

    args = arg_parser.parse_args()

    # main(
    #     input_dataset_file=Path('grammar_generation/data/paraphrase.debug.jsonl'),
    #     output_file=Path('grammar_generation/data/all_derives_scholar_6.234402f7.train.template.k1000.paraphrase_augmented.jsonl'),
    #     model_path=Path('data/paraphrase/checkpoints/checkpoint_best.pt'),
    #     beam_size=30,
    #     batch_size=1,
    #     constrained_decoding=True
    # )
    main(**args.__dict__)

    # for beam_size in [5, 10, 20]:
    #     data_file = Path('grammar_generation/data/all_derives_scholar_6.c07001cc.train.template.k1000.template_split.train.jsonl')
    #
    #     main(
    #         input_dataset_file=data_file,
    #         output_file=data_file.with_suffix(f'.augmented.0105.bs{beam_size}.cd.jsonl'),
    #         model_path=Path(
    #             'data/paraphrase/checkpoints-maxtok512-updatefreq1-iter10000-warmup1000/checkpoint_best.pt'),
    #         beam_size=beam_size,
    #         batch_size=1,
    #         constrained_decoding=True
    #     )
    #
    #     main(
    #         input_dataset_file=data_file,
    #         output_file=data_file.with_suffix('.augmented.0105.bs10.jsonl'),
    #         model_path=Path(
    #             'data/paraphrase/checkpoints-maxtok512-updatefreq1-iter10000-warmup1000/checkpoint_best.pt'),
    #         beam_size=beam_size,
    #         batch_size=10,
    #         constrained_decoding=False
    #     )
