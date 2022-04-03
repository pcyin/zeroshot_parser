#!/bin/bash
set -e
set -x

echo ${WORK_DIR}

PYTHONPATH=. python grammar_generation/sub_sample_dev_data.py \
    --action collect \
    --paraphrase-tree-path ${WORK_DIR}/paraphrase_tree.train.jsonl

python grammar_generation/lm_scoring.py \
    --dataset ${WORK_DIR}/paraphrase_tree.train.valid_examples.jsonl \
    --output ${WORK_DIR}/paraphrase_tree.train.valid_examples.lm_score.txt \
    --cuda \
    --num-workers=1 \
    --batch-size 64 \
    --model-name gpt2-xl

for split_method in nat_iid_sample nat_template_sample iid_sample
do
  for k in 2000 4000
  do
    PYTHONPATH=. python grammar_generation/sub_sample_dev_data.py \
      --action sample \
      --seed-dataset-path ${WORK_DIR}/paraphrase_tree.train.valid_examples.jsonl \
      --lm-score-path ${WORK_DIR}/paraphrase_tree.train.valid_examples.lm_score.txt \
      --output ${WORK_DIR}/dev.${split_method}.k${k}.jsonl \
      --k=$k  \
      --strategy ${split_method}
  done
done
