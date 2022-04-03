import gc
import itertools
import time
from pathlib import Path
from typing import List, Dict

from allennlp.common import Registrable
from allennlp.models import Model, Archive
import torch
import torch.nn as nn
import faiss
import numpy as np
from tqdm import tqdm

from nsp.dataset_readers.seq2seq_with_copy_reader import SequenceToSequenceModelWithCopyReader
from paraphrase.paraphrase_identification import run_paraphrase_identification_model
from paraphrase.paraphrase_tree import ParaphraseTree, Example
from paraphrase.sim.sim_utils import SimModel
from paraphrase.sim.test_sim import encode_sentences
from paraphrase.utils import get_batches

from nsp.models.seq2seq_with_copy import Seq2SeqModelWithCopy


def normalize_lf(lf_string: str) -> str:
    return ' '.join([
        x
        for x
        in lf_string.strip().split(' ')
        if x not in {'(', ')', 'name'}
    ])


@torch.no_grad()
def build_index(examples: List[Example], sim_model: SimModel) -> Dict:
    # index sentences
    print('Indexing sentences...')

    from grammar_generation.generate_dataset_from_sampled_examples import PLACEHOLDER_ENTITY_MAP
    named_entities = sorted(PLACEHOLDER_ENTITY_MAP)
    typed_slot_to_named_entity_map = {}

    for named_entity in named_entities:
        entry = PLACEHOLDER_ENTITY_MAP[named_entity]
        typed_slot_name = f'{entry["type_name"]}0'
        alternative_typed_slot_name = f'{entry["type_name"]}1'

        if typed_slot_name not in typed_slot_to_named_entity_map:
            typed_slot_to_named_entity_map[typed_slot_name] = named_entity
        else:
            typed_slot_to_named_entity_map[alternative_typed_slot_name] = named_entity

    # FIXME: this is buggy!!!!
    def normalize_utterance(utterance: str) -> str:
        tokens = utterance.split()
        normalized_tokens = []
        for token in tokens:
            normalized_tokens.append(typed_slot_to_named_entity_map.get(token, token))

        return ' '.join(normalized_tokens)

    sentences = []
    sentence_id_to_example_idx = {}
    example_idx_to_sentence_id = {}
    for example in examples:
        sentence_id_to_example_idx[len(sentences)] = example['idx']
        example_idx_to_sentence_id[example['idx']] = len(sentences)

        sentences.append(normalize_utterance(example['can']))

    batches = get_batches(sentences, batch_size=64)
    sentence_vec_tensors = [
        encode_sentences(batch, sim_model)
        for batch
        in batches
    ]
    sentence_vecs = torch.cat(sentence_vec_tensors, dim=0).cpu().numpy()
    sentence_vecs = sentence_vecs / np.linalg.norm(sentence_vecs, axis=-1, keepdims=True)
    # print(sentence_vecs[0])
    assert np.abs(np.square(sentence_vecs[0]).sum() - 1.0) <= 1e-5

    print('[Pruner] building index...')
    index = faiss.IndexFlatIP(300)
    index.add(sentence_vecs)

    if torch.cuda.is_available():
        print(print('[Pruner] building index on GPU...'))
        index = faiss.index_cpu_to_gpus_list(index, ngpu=1)

    return dict(
        index=index,
        sentence_id_to_example_idx=sentence_id_to_example_idx,
        example_idx_to_sentence_id=example_idx_to_sentence_id,
        sentence_vecs=sentence_vecs
    )


@torch.no_grad()
def prune_using_parser_score(
    paraphrase_tree: ParaphraseTree,
    parser_archive: Archive,
    sim_model: SimModel,
    examples: List[Example],
    iter_idx: int,
    K: int = 5,
    index_all_descendants: bool = False
):
    if index_all_descendants:
        examples_to_index = []
        for depth in range(1, paraphrase_tree.depth):
            examples_to_index.extend([
                e
                for e
                in paraphrase_tree.get_examples_by_level(depth)
                if e['is_valid_paraphrase']
            ])
    else:
        examples_to_index = examples

    index_dict = build_index(examples_to_index, sim_model)
    index = index_dict['index']
    sentence_vecs = index_dict['sentence_vecs']
    example_idx_to_sentence_id = index_dict['example_idx_to_sentence_id']
    sentence_id_to_example_idx = index_dict['sentence_id_to_example_idx']

    parser_dataset_reader: SequenceToSequenceModelWithCopyReader = parser_archive.dataset_reader  # noqa
    parser: Seq2SeqModelWithCopy = parser_archive.model  # noqa

    num_false_positives = 0
    num_prev_positive_examples = 0
    t1 = time.time()

    pbar = tqdm(total=len(examples), desc='Pruning...')
    for batched_examples in get_batches(examples, batch_size=64):
        examples_batch_info = dict()
        for e_id, example in enumerate(batched_examples):
            is_valid_paraphrase = example['is_valid_paraphrase']
            if not is_valid_paraphrase:
                continue

            # if any of its parents are pruned, we should also prune this node
            example_idx = example['idx']
            parent_example = paraphrase_tree.get_parent_example_by_idx(example_idx)
            pruned_by_parent = False
            while parent_example:
                if 'paraphrase' in str(parent_example['idx']) and not parent_example['is_valid_paraphrase']:
                    pruned_by_parent = True
                    break
                parent_example = paraphrase_tree.get_parent_example_by_idx(parent_example['idx'])

            if pruned_by_parent:
                num_prev_positive_examples += 1
                num_false_positives += 1

                example['is_valid_paraphrase'] = False
                pruner_metadata = {'is_false_positive': True, 'pruned_by_parent': True}
                example.setdefault('pruner_metadata', {})[iter_idx] = pruner_metadata
                continue

            vec_id = example_idx_to_sentence_id[example_idx]
            example_vec = sentence_vecs[vec_id]
            D, I = index.search(example_vec[None, :], k=100)
            nbr_idx_list = [
                (sentence_id_to_example_idx[idx], float(sim_score))
                for idx, sim_score
                in zip(list(I[0]), D[0])
            ]

            valid_nbr_idx_list = []
            nbr_ptr = 0
            example_variables = set(example['variables'])
            while len(valid_nbr_idx_list) < K and nbr_ptr < len(nbr_idx_list):
                nbr_idx, sim_score = nbr_idx_list[nbr_ptr]
                nbr_example = paraphrase_tree.get_example_by_id(nbr_idx)
                if (
                    normalize_lf(nbr_example['canonical_lf']) != normalize_lf(example['canonical_lf']) and
                    set(nbr_example['variables']) == example_variables
                ):
                    valid_nbr_idx_list.append((nbr_idx, sim_score))

                nbr_ptr += 1

            if valid_nbr_idx_list:
                batch_instances = []
                example_inst = parser_dataset_reader.text_to_instance(
                    source_sequence=example['nl'],
                    target_sequence=example['lf']
                )
                batch_instances.append(example_inst)

                for nbr_example_idx, sim_score in valid_nbr_idx_list:
                    nbr_example = paraphrase_tree.get_example_by_id(nbr_example_idx)
                    nbr_inst = parser_dataset_reader.text_to_instance(
                        source_sequence=example['nl'],
                        target_sequence=nbr_example['lf']
                    )
                    batch_instances.append(nbr_inst)

                examples_batch_info[e_id] = {
                    'nbr_idx': [x[0] for x in valid_nbr_idx_list],
                    'nbr_sim_score': [x[1] for x in valid_nbr_idx_list],
                    'instances': batch_instances
                }

        parser_insts = list(itertools.chain.from_iterable(
            examples_batch_info.get(e_id, {}).get('instances', [])
            for e_id
            in range(len(batched_examples))
        ))

        inst_log_probs = []
        for batch in get_batches(parser_insts, batch_size=64):
            batch_outputs = parser.forward_on_instances(batch)
            batch_log_probs: List[float] = [float(x['example_log_probs']) for x in batch_outputs]
            inst_log_probs.extend(batch_log_probs)

        inst_ptr = 0
        for e_id, example in enumerate(batched_examples):
            if e_id in examples_batch_info:
                valid_nbr_idx_list = examples_batch_info[e_id]['nbr_idx']
                log_probs = inst_log_probs[inst_ptr: inst_ptr + len(examples_batch_info[e_id]['instances'])]
                inst_ptr += len(examples_batch_info[e_id]['instances'])

                pruner_metadata = {
                    'nbr_idx': valid_nbr_idx_list,
                    'nbr_sim_score': examples_batch_info[e_id]['nbr_sim_score'],
                    'example_log_prob': log_probs[0],
                    'nbr_log_probs': log_probs[1:]
                }

                is_valid_paraphrase = example['is_valid_paraphrase']
                # check if p(lf|x) > p(nbr_lf|x)
                for idx, nbr_example_idx in enumerate(valid_nbr_idx_list):
                    p_lf_x = log_probs[0]
                    p_nbr_lf_x = log_probs[1 + idx]

                    if p_nbr_lf_x > p_lf_x:
                        is_valid_paraphrase = False
                        break

                was_valid_paraphrase = example['is_valid_paraphrase']
                example['is_valid_paraphrase'] = is_valid_paraphrase

                if was_valid_paraphrase:
                    num_prev_positive_examples += 1

                    if not is_valid_paraphrase:
                        num_false_positives += 1
                        pruner_metadata['is_false_positive'] = True

                example.setdefault('pruner_metadata', {})[iter_idx] = pruner_metadata

        pbar.update(len(batched_examples))

    filtered_examples = [
        example
        for example in examples
        if example['is_valid_paraphrase']
    ]

    t2 = time.time()
    print(f'[Pruner] Removed {num_false_positives}/{num_prev_positive_examples}={num_false_positives / num_prev_positive_examples} false positives (took {t2 - t1:.1f}s)')

    return filtered_examples


class ParaphrasePruner:
    def prune(self, examples: List[Example], iter_idx: int, **kwargs) -> List[Example]:
        raise NotImplementedError


@Registrable.register('parser_score_pruner')
class ParserScoreParaphrasePruner(ParaphrasePruner):
    def __init__(self, paraphrase_tree: ParaphraseTree, parser_archive: Archive, sim_model: SimModel, K: int = 5, index_all_descendants: bool = False):
        self.paraphrase_tree = paraphrase_tree
        self.parser_archive = parser_archive
        self.sim_model = sim_model
        self.K = K
        self.index_all_descendants = index_all_descendants

        self.parser_archive.model.eval()
        self.parser_archive.model._prediction = False

    def prune(self, examples: List[Example], iter_idx: int) -> List[Example]:
        with torch.no_grad():
            return prune_using_parser_score(
                self.paraphrase_tree,
                self.parser_archive,
                self.sim_model,
                examples,
                iter_idx,
                K=self.K,
                index_all_descendants=self.index_all_descendants
            )


@Registrable.register('identification_score_pruner')
class ParaphraseIdentificationModelPruner(ParaphrasePruner):
    def __init__(self, paraphrase_tree: ParaphraseTree, sim_model: SimModel, paraphrase_identification_model_path: Path, K: int = 5, index_all_descendants: bool = False):
        self.K = K
        self.sim_model = sim_model
        self.paraphrase_tree = paraphrase_tree
        self.paraphrase_identification_model_path = paraphrase_identification_model_path
        self.index_all_descendants = index_all_descendants

    def prune(self, examples: List[Example], iter_idx: int) -> List[Example]:
        if self.index_all_descendants:
            examples_to_index = []
            for depth in range(1, self.paraphrase_tree.depth):
                examples_to_index.extend([
                    e
                    for e
                    in self.paraphrase_tree.get_examples_by_level(depth)
                    if e['is_valid_paraphrase']
                ])
        else:
            examples_to_index = examples

        with torch.no_grad():
            index_dict = build_index(examples_to_index, self.sim_model)
            index = index_dict['index']
            sentence_vecs = index_dict['sentence_vecs']
            example_idx_to_sentence_id = index_dict['example_idx_to_sentence_id']
            sentence_id_to_example_idx = index_dict['sentence_id_to_example_idx']

            pi_instances = []
            metadata_idx = {}
            t1 = time.time()
            for example in tqdm(examples, desc='Generating Paraphrase Identification Model Examples...'):
                is_valid_paraphrase = example['is_valid_paraphrase']
                if not is_valid_paraphrase:
                    continue

                vec_id = example_idx_to_sentence_id[example['idx']]
                example_vec = sentence_vecs[vec_id]
                D, I = index.search(example_vec[None, :], k=min(100, len(examples)))
                nbr_idx_list = [
                    sentence_id_to_example_idx[idx]
                    for idx
                    in list(I[0])
                ]

                valid_nbr_idx_list = []
                nbr_ptr = 0
                while len(valid_nbr_idx_list) < self.K and nbr_ptr < len(nbr_idx_list):
                    nbr_idx = nbr_idx_list[nbr_ptr]
                    nbr_example = self.paraphrase_tree.get_example_by_id(nbr_idx)
                    if normalize_lf(nbr_example['canonical_lf']) != normalize_lf(example['canonical_lf']):
                        valid_nbr_idx_list.append(nbr_idx)

                    nbr_ptr += 1

                def is_valid_nbr_example(example: Example, nbr_example: Example) -> bool:
                    return set(example['variables']) == set(nbr_example['variables'])

                valid_nbr_idx_list = [
                    nbr_idx
                    for nbr_idx
                    in valid_nbr_idx_list
                    if is_valid_nbr_example(
                        example,
                        self.paraphrase_tree.get_example_by_id(nbr_idx)
                    )
                ]

                pruner_metadata = {'nbr_idx': valid_nbr_idx_list}
                metadata_idx[example['idx']] = pruner_metadata

                if valid_nbr_idx_list:
                    example_inst = {
                        'idx': f'{example["idx"]}:self',
                        'sent_1': self.paraphrase_tree.get_parent_example_by_idx(example['idx'])['can'],
                        'sent_2': example['can']
                    }
                    pi_instances.append(example_inst)

                    for nbr_example_idx in valid_nbr_idx_list:
                        nbr_example = self.paraphrase_tree.get_example_by_id(nbr_example_idx)
                        nbr_parent_example = self.paraphrase_tree.get_parent_example_by_idx(nbr_example_idx)
                        nbr_inst = {
                            'idx': f'{example["idx"]}:{nbr_example["idx"]}',
                            'sent_1': nbr_parent_example['can'],
                            'sent_2': example['can']
                        }
                        pi_instances.append(nbr_inst)

            t2 = time.time()
            print(f'[Pruner] {t2 - t1:.1f}s took to generate examples')

            del index
            gc.collect()

            t1 = time.time()
            predictions = run_paraphrase_identification_model(
                pi_instances, self.paraphrase_identification_model_path, batch_size=256)
            t2 = time.time()
            print(f'[Pruner] {t2 - t1:.1f}s took to perform inference using the paraphrase ID model')

            num_false_positives = 0
            num_prev_positive_examples = 0
            t1 = time.time()
            for example in tqdm(examples, desc='Pruning...'):
                is_valid_paraphrase = example['is_valid_paraphrase']
                if not is_valid_paraphrase:
                    continue

                pruner_metadata = metadata_idx[example['idx']]
                valid_nbr_idx_list = pruner_metadata['nbr_idx']

                if valid_nbr_idx_list:
                    # check if p(lf|x) > p(nbr_lf|x)
                    example_idx = example['idx']
                    pruner_metadata['p_x_parent_and_x'] = p_x_parent_and_x = float(predictions[f'{example_idx}:self']['prob'][1])
                    pruner_metadata['p_nbr_parent_and_x'] = []

                    for idx, nbr_example_idx in enumerate(valid_nbr_idx_list):
                        p_nbr_parent_x_and_x = float(predictions[f'{example_idx}:{nbr_example_idx}']['prob'][1])
                        pruner_metadata['p_nbr_parent_and_x'].append(p_nbr_parent_x_and_x)
                        if p_nbr_parent_x_and_x > p_x_parent_and_x:
                            is_valid_paraphrase = False

                was_valid_paraphrase = example['is_valid_paraphrase']
                example['is_valid_paraphrase'] = is_valid_paraphrase

                if was_valid_paraphrase:
                    num_prev_positive_examples += 1

                if was_valid_paraphrase and not is_valid_paraphrase:
                    num_false_positives += 1
                    pruner_metadata['is_false_positive'] = True

                example.setdefault('pruner_metadata', {})[iter_idx] = pruner_metadata

        filtered_examples = [
            example
            for example in examples
            if example['is_valid_paraphrase']
        ]
        t2 = time.time()

        print(f'[Pruner] Removed {num_false_positives}/{num_prev_positive_examples}={num_false_positives / num_prev_positive_examples} false positives (took {t2-t1:.1f}s)')

        return filtered_examples


if __name__ == '__main__':
    pass
