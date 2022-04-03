from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass, ChoiceEnum


class Config(FairseqDataclass):
    def to_dict(self) -> Dict:
        return {
            k: getattr(self, k)
            for k in self._get_all_attributes()
        }


@dataclass
class ParserConfig(Config):
    batch_size: int = field(default=False, metadata={'help': 'batch size'})
    validation_metric: ChoiceEnum(['seq_acc', 'ppl']) = field(default='seq_acc', metadata={'help': ''})
    logical_form_data_field: str = field(default='lf')
    only_use_parser_filtered_paraphrase_example: bool = field(default=False)
    use_canonical_example: bool = field(default=True)
    do_eval: bool = field(default=True)
    use_cumulated_datasets: bool = field(default=False)
    rerun_prediction: bool = field(default=False)
    tgt_stopword_loss_weight: Optional[float] = field(default=None)
    validation_after_epoch: int = field(default=0)
    max_epoch: int = field(default=30)
    patience: int = field(default=20)
    dropout: float = field(default=0.2)
    config_file: Path = field(default=Path('configs/config_bert.json'))
    from_pretrained: Path = field(default=None)


@dataclass
class ParaphraserConfig(Config):
    batch_size: int = field(default=64)
    parser_batch_size: int = field(default=64)
    beam_size: int = field(default=10)
    include_statement: bool = field(default=False)
    include_question: bool = field(default=True)
    sampling: bool = field(default=False)
    sampling_topp: float = field(default=0.)
    include_source_examples: bool = field(default=False)
    filter_example_by_sim_score_threshold: float = field(default=0.)
    paraphrase_dev_set: bool = field(default=True)
    seed_file_type: str = field(default='filtered')
    heuristic_deduplicate: bool = field(default=False)
    do_not_filter_train_set_using_parser: bool = field(default=False)
    parser_allowed_rank_in_beam: int = field(default=1)
    lm_scorer: str = field(default=None)
    extra_config_string: str = field(default='')


@dataclass
class ParaphraseIdentificationModelConfig(Config):
    enabled: bool = field(default=False)
    process_dev_data: bool = field(default=True)
    only_use_parser_accepted_examples: bool = field(default=False)
    model_name_or_path: str = field(default=None)
    batch_size: int = field(default=32)
    lr: float = field(default=2e-5)
    epoch: float = field(default=2)
    sample_size: Optional[int] = field(default=None)
    sample_negative_example_by_sim_score: bool = field(default=True)
    sample_positive_example_by_sim_score: bool = field(default=True)
    num_folds: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)

    use_pruner: bool = field(default=False)
    pruner_name: str = field(default=None, metadata={'choices': ['parser_score_pruner', 'identification_score_pruner']})
    pruner_nbr_num: int = field(default=5)
    pruner_index_all_descendants: bool = field(default=False)
    use_model_labeled_positive_examples: bool = field(default=False)

    inference_only: bool = field(default=False)

    oracle_experiment: bool = field(default=False)
    oracle_experiment_label_strategy: str = field(default='majority')
    oracle_experiment_dev_split_ratio: float = field(default=0.)
