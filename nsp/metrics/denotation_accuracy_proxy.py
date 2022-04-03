import json
import math
from typing import List, Dict, Any

from overrides import overrides

from allennlp.training.metrics import Metric

import tempfile
import subprocess
import os
import time

from evaluator.executor_client import ExecutorClient
from nsp.metrics.denotation_accuracy import lexicalize_entity, format_lf, rep_to_empty_set, is_error, pick_derivations


@Metric.register("denotation_accuracy_proxy")
class DenotationAccuracyProxy(Metric):
    """
    Calculates the denotation accuracy.
    """

    def __init__(
        self,
        executor_addr: str = 'http://localhost:8081/'
    ) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.
        self._true_answers = []
        self._preds = []
        self._vars = []
        self._indices = []
        self.executor = ExecutorClient(executor_addr)

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.
        self._true_answers = []
        self._preds = []
        self._indices = []

    @overrides
    def __call__(
        self,
        predictions: List[List[List[str]]],
        gold_targets: List[List[str]],
        variables: List[dict],
        indices: List[int]
    ) -> None:
        self._total_counts += len(predictions)
        self._vars.append(variables)
        for predicted_tokens, gold_tokens, var, index in zip(predictions, gold_targets, variables, indices):

            true_answer_rep = []
            for i, t in enumerate(gold_tokens):
                next_token = lexicalize_entity(var, t)

                if t.startswith('misc') and 'number' in next_token and i > 0 and gold_tokens[i - 1] != '(':
                    next_token = '( ' + next_token + ' )'

                true_answer_rep.append(next_token)

            true_answer_lf = " ".join(true_answer_rep)
            self._true_answers.append(true_answer_lf)
            self._indices.append(index)

            preds = []
            for pred in predicted_tokens:

                pred_rep = []
                for i, t in enumerate(pred):
                    next_token = lexicalize_entity(var, t)

                    if t.startswith('misc') and 'number' in next_token and i > 0 and pred[i - 1] != '(':
                        next_token = '( ' + next_token + ' )'

                    pred_rep.append(next_token)

                pred_lf = " ".join(pred_rep)
                preds.append(pred_lf)
            self._preds.append(preds)

    def get_denotations(self, lfs: List[str]):
        response = self.executor.execute(lfs)

        assert all(
            line.startswith('targetValue')
            for line
            in response['results']
        )
        denotations = [
            line.strip().split("\t")[1]
            for line
            in response['results']
        ]

        return denotations

    def request(self, all_lfs: List[str], batch_size: int = 100) -> List[Any]:
        denotations = []
        batch_num = int(math.ceil(len(all_lfs) / batch_size))

        for i in range(batch_num):
            batch = all_lfs[batch_size * i: batch_size * (i + 1)]
            batch_denotations = self.get_denotations(batch)
            denotations.extend(batch_denotations)

        return denotations

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:

        accuracy = 0.
        ind_to_match = None

        all_lfs = ([format_lf(s) for s in self._true_answers] +
                   [format_lf(s) for p in self._preds for s in p])

        if reset and all_lfs:
            denotations = self.request(all_lfs)

            assert len(denotations) == len(all_lfs)

            true_dens = denotations[:len(self._true_answers)]
            unmodified_true_dens = list(true_dens)
            all_pred_dens = denotations[len(self._true_answers):]

            all_pred_dens_ = []
            for i, pred_d in enumerate(all_pred_dens):
                all_pred_dens_.append(rep_to_empty_set(pred_d))
            all_pred_dens = all_pred_dens_
            true_dens = [rep_to_empty_set(pred_d) for pred_d in true_dens]

            derivs, pred_dens = pick_derivations(all_pred_dens, self._preds, is_error)
            match = [t == p and not is_error(t) for t, p in zip(true_dens, pred_dens)]

            for idx, (e_idx, true_denotation) in enumerate(zip(self._indices, unmodified_true_dens)):
                if is_error(true_denotation):
                    print(f'WARNING: Error in executing the target logical form of example {e_idx}: {self._true_answers[idx]}')
                    print('Result:', true_denotation)
                if rep_to_empty_set(true_denotation) != true_denotation:
                    print(f'Converting the error result of example {e_idx} to empty list')

            ind_to_match = dict()
            for (i, m) in zip(self._indices, match):
                ind_to_match[i] = m

            accuracy = sum(match) / len(match)
            self.reset()

        return {
            "den_acc": accuracy,
            'ind_to_match': ind_to_match
        }


if __name__ == '__main__':
    metric = DenotationAccuracyProxy()

    examples = [
        json.loads(line)
        for line in open('data/test_scholar.json')
    ]

    metric(
        [[e['lf'].split()] for e in examples],
        [e['lf'].split() for e in examples],
        [e['variables'] for e in examples],
        [1 for e in examples]
    )

    print(metric.get_metric(reset=True))
