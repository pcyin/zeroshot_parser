#!/bin/bash
#SBATCH --mem=24g
#SBATCH --time=0
#SBATCH --cpus=10

rm module-classes.txt

. /projects/tir1/users/pengchey/anaconda2/etc/profile.d/conda.sh

conda activate data_efficient_parsing
which python

DOMAIN="geo"

PREFIX=all_derives_${DOMAIN}_6.84653bf2.rerun.norule_total.norule_stateofcity

PYTHONPATH=. python grammar_generation/grammar_gen.py \
  --max_depth=6 \
  --domain ${DOMAIN} \
  --domain-grammar grammars/geo880.natural.84653bf2.grammar \
  --disabled-rules '$NP -> total $RelNP of $NP' \
  --output_file_prefix=grammar_generation/data/$PREFIX
