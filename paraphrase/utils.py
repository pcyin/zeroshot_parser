import math
import re
import string
from collections import Counter
from typing import List, Any, Iterator, Dict, Tuple, Set, Union

import torch
from scipy.stats import kendalltau
from spacy.tokens import Token
from spacy.tokens.doc import Doc

from paraphrase.sim.test_sim import find_similarity


def get_batches(
    dataset: List[Any],
    batch_size: int
) -> Iterator[List[Any]]:
    batch_num = math.ceil(len(dataset) / batch_size)
    for i in range(batch_num):
        examples = dataset[i * batch_size: (i + 1) * batch_size]

        yield examples


def decanonicalize_utterance(utterance: str, variable_info: Dict):
    for canonical_entity_name, entry in variable_info.items():
        utterance = utterance.replace(canonical_entity_name, entry['mention'].lower())

    return utterance


def get_paraphrase_tree(examples: List[Dict]) -> Dict:
    paraphrase_tree = dict()
    for example in examples:
        m = re.match(r'^(.*)-paraphrase-(\d+)$', example['idx'])
        idx_prefix = m.group(1)
        paraphrase_idx = int(m.group(2))
        if 'paraphrase' in idx_prefix:
            pass

        # original_src_idx, _, paraphrase_id_str = example['idx'].partition('-')
        # paraphrase_idx = re.match(r'paraphrase-(\d+)', paraphrase_id_str).group(1)

        parent_utterance = decanonicalize_utterance(example['source_nl'], example['variable_info'])

        paraphrase_tree.setdefault(
            idx_prefix,
            dict(can=parent_utterance, children=dict())
        )['children'][paraphrase_idx] = example

    return paraphrase_tree


@torch.no_grad()
def get_sim_scores(examples: List[Tuple], batch_size: int, model: Dict) -> List[float]:
    total_batches = len(examples) // batch_size

    all_scores = []
    for batch in (
        get_batches(examples, batch_size=batch_size)
        #desc='Computing SIM score...', total=total_batches
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

    return all_scores


def extract_ngrams(tokens: List[Union[Token, str]], n: int) -> List[str]:
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tokens[i: i + n]
        if isinstance(tokens[0], Token):
            ngram_norm_text = ' '.join([token.lemma_.lower() for token in ngram])
        else:
            ngram_norm_text = ' '.join(ngram)

        ngrams.append(ngram_norm_text)

    return ngrams


def get_ngram_overlap(ngrams_a: Set, ngrams_b: Set) -> float:
    return len(ngrams_a.intersection(ngrams_b)) / len(ngrams_a.union(ngrams_b))


# same as in john weiting's paper
def get_trigram_overlap(ngrams_a: Set, ngrams_b: Set) -> float:
    return len(ngrams_a.intersection(ngrams_b)) / min(len(ngrams_a), len(ngrams_b))


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Calculate word level F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0, 1.0, 1.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def get_kendall_tau(x1, x2):
    x1 = normalize_answer(x1)
    x2 = normalize_answer(x2)

    x1_tokens = x1.split()
    x2_tokens = x2.split()

    for x1_index, tok in enumerate(x1_tokens):
        try:
            x2_index = x2_tokens.index(tok)
            x1_tokens[x1_index] = "<match-found>-{:d}".format(x1_index + 1)
            x2_tokens[x2_index] = "<match-found>-{:d}".format(x1_index + 1)
        except ValueError:
            pass

    common_seq_x1 = [int(x1_tok_flag.split("-")[-1]) for x1_tok_flag in x1_tokens if x1_tok_flag.startswith("<match-found>")]
    common_seq_x2 = [int(x2_tok_flag.split("-")[-1]) for x2_tok_flag in x2_tokens if x2_tok_flag.startswith("<match-found>")]

    assert len(common_seq_x1) == len(common_seq_x2)

    ktd = kendalltau(common_seq_x1, common_seq_x2).correlation
    anomaly = False

    if math.isnan(ktd):
        ktd = -1.0
        anomaly = True

    return ktd, anomaly


def normalize_utterance(utterance: Doc):
    norm_tokens = [tok.lemma_.lower() for tok in utterance if tok.text.lower() not in ['the', 'a', 'an']]
    return ' '.join(norm_tokens)
