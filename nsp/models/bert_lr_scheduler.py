from collections import defaultdict
from typing import List, Tuple, Dict, Any

import transformers
from allennlp.training.learning_rate_schedulers import PolynomialDecay
from allennlp.training.optimizers import Optimizer
from overrides import overrides
import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("bert2seq")
class Bert2SeqLearningRateScheduler(LearningRateScheduler):
    """
    Implements polynomial decay Learning rate scheduling. The learning rate is first
    linearly increased for the first `warmup_steps` training steps. Then it is decayed for
    `total_steps` - `warmup_steps` from the initial learning rate to `end_learning_rate` using a polynomial
    of degree `power`.

    Formally,

    `lr` = (`initial_lr` - `end_learning_rate`) *
           ((`total_steps` - `steps`)/(`total_steps` - `warmup_steps`)) ** `power`

    # Parameters

    total_steps: `int`, required
        The total number of steps to adjust the learning rate for.
    warmup_steps : `int`, required
        The number of steps to linearly increase the learning rate.
    power : `float`, optional (default = `1.0`)
        The power of the polynomial used for decaying.
    end_learning_rate : `float`, optional (default = `0.0`)
        Final learning rate to decay towards.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_steps_per_epoch: int,
        freeze_bert_first_num_steps: int,
        power=1.0,
        warmup_steps=0,
        end_learning_rate=0.0,
        schedule: str = None,
        auto_warmup: bool = False,
        last_epoch: int = -1,
        reset_optim: bool = False
    ):
        super().__init__(optimizer, last_epoch)

        self.is_reset = False
        self.reset_optim = reset_optim

        self.power = power
        total_steps = num_epochs * num_steps_per_epoch

        if schedule:
            print(f'Use schedule {schedule}')
            for i, x in enumerate(schedule.split(',')):
                if x != 'null':
                    r = float(x)
                    if i == 0:
                        freeze_bert_first_num_steps = int(total_steps * r)
                    elif i == 1:
                        warmup_steps = int(total_steps * r)

        assert total_steps > freeze_bert_first_num_steps
        self.freeze_bert_first_num_steps = freeze_bert_first_num_steps
        self.total_steps = total_steps - freeze_bert_first_num_steps
        self.warmup_steps = warmup_steps if not auto_warmup else int(self.total_steps * 0.1)

        print(f'Bert Freeze first steps: {freeze_bert_first_num_steps}; Warm up steps: {self.warmup_steps}; Total steps: {self.total_steps}')

        self.end_learning_rate = end_learning_rate

        self.steps = 0

        self.param_group_weight_decays = [p['weight_decay'] for p in self.optimizer.param_groups]
        for p in self.optimizer.param_groups:
            p['weight_decay'] = 0.00

        self.step_batch(0)

    @overrides
    def get_values(self):
        if self.is_reset:
            steps = self.steps - self.freeze_bert_first_num_steps
            warmup_steps = self.warmup_steps
        else:
            warmup_steps = self.freeze_bert_first_num_steps
            steps = self.steps

        if warmup_steps > 0 and steps < warmup_steps:
            f = steps / warmup_steps
            lrs = [
                f * lr
                for lr
                in self.base_values
            ]

        elif steps >= self.total_steps:
            lrs = [self.end_learning_rate for _ in self.base_values]

        else:
            current_decay_steps = self.total_steps - steps
            total_decay_steps = self.total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps) ** self.power

            lrs = [
                f * (lr - self.end_learning_rate) + self.end_learning_rate for lr in self.base_values
            ]

        if not self.is_reset:
            for i in range(len(lrs) - 1):
                lrs[i] = 0.

            lrs[-1] = 0.001

        print(lrs)

        return lrs

    @overrides
    def step(self, metric: float = None) -> None:
        pass

    @overrides
    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total is None:
            self.steps += 1
        else:
            self.steps = batch_num_total

        if self.steps >= self.freeze_bert_first_num_steps and not self.is_reset:
            print('Unfreeze BERT. Reset LR.')
            self.is_reset = True
            if self.reset_optim:
                print('Reset optimizer state')
                self.optimizer.__setstate__({'state': defaultdict(dict)})

            for i, p in enumerate(self.optimizer.param_groups):
                p['weight_decay'] = self.param_group_weight_decays[i]

        assert len(self.optimizer.param_groups) == 3

        for param_group, lr in zip(self.optimizer.param_groups, self.get_values()):
            # print(len(param_group['params']), lr)
            param_group[self.param_group_field] = lr


@Optimizer.register("bert2seq_adamw")
class Bert2SeqAdamWOptimizer(Optimizer, transformers.AdamW):
    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        bert_params = [
            (p_name, p)
            for p_name, p in model_parameters
            if p_name.startswith('_source_embedder') and p.requires_grad
        ]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        other_params = [
            (p_name, p)
            for p_name, p
            in model_parameters
            if not p_name.startswith('_source_embedder') and p.requires_grad
        ]

        print(f'Num. BERT Params: {len(bert_params)}, Num. other Params: {len(other_params)}')

        grouped_parameters = [
            {
                'params': [
                    p
                    for p_name, p
                    in bert_params
                    if not any(name in p_name for name in no_decay)
                ],
                'weight_decay': weight_decay
            },
            {
                'params': [
                    p
                    for p_name, p
                    in bert_params
                    if any(name in p_name for name in no_decay)
                ],
                'weight_decay': 0.00
            },
            {
                'params': [
                    p
                    for p_name, p
                    in other_params
                ],
                'weight_decay': 0.00,
                'lr': 0.001
            }
        ]

        super().__init__(
            params=grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
