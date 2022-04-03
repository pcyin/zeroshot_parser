#!/bin/bash
set +e

if [ -f "module-classes.txt" ] ; then
    rm "module-classes.txt"
fi

DOMAIN="scholar" # or "geo"

MAX_DEPTH=6  # Reduce this number for debugging.
PREFIX=all_derives_${DOMAIN}_${MAX_DEPTH}

echo "Sample canonical examples from SCFG."
if [ ${DOMAIN} == "scholar" ]
then
  PYTHONPATH=. python grammar_generation/grammar_gen.py \
    --max_depth=${MAX_DEPTH} \
    --domain ${DOMAIN} \
    --disabled-rules '$NP -> total $RelNP of $NP' \
    --output_file_prefix=grammar_generation/data/$PREFIX
else
  PYTHONPATH=. python grammar_generation/grammar_gen.py \
  --max_depth=${MAX_DEPTH} \
  --domain ${DOMAIN} \
  --disabled-rules '$NP -> total $RelNP of $NP' '$RelNP -> state' '$EntityNP1 -> 10' \
  --output_file_prefix=grammar_generation/data/$PREFIX
fi

echo "Generating LM scores."
PYTHONPATH=. \
python grammar_generation/lm_scoring.py \
    --dataset "grammar_generation/data/${PREFIX}.pruned_lf.txt" \
    --output "grammar_generation/data/ ${PREFIX}.pruned_lf.lm_score.txt" \
    --batch-size 128 \
    --model-name gpt2-xl \
    --num-workers=4 \
    --cuda
