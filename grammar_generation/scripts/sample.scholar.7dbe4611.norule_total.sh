#!/bin/bash
#SBATCH --exclude=compute-0-[3,26,31,36],compute-1-[7,13,18]
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --cpus=10

rm module-classes.txt

. /projects/tir1/users/pengchey/anaconda2/etc/profile.d/conda.sh

conda activate data_efficient_parsing
which python

DOMAIN="scholar"

PREFIX=all_derives_${DOMAIN}_6.7dbe4611.norule_total

PYTHONPATH=. python grammar_generation/grammar_gen.py \
  --max_depth=6 \
  --domain ${DOMAIN} \
  --disabled-rules '$NP -> total $RelNP of $NP' \
  --output_file_prefix=grammar_generation/data/$PREFIX
