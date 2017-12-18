#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3
GPU=$4
USE_EXISTING=$5

python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=checkpoints/ens3_adv_inception_v3_rename.ckpt \
  --checkpoint_path_adv_inception_v3=checkpoints/adv_inception_v3_rename.ckpt \
  --checkpoint_path_ens3_adv_inception_v3=checkpoints/ens3_adv_inception_v3_rename.ckpt \
  --checkpoint_path_ens4_adv_inception_v3=checkpoints/ens4_adv_inception_v3_rename.ckpt \
  --checkpoint_path_inception_v4=checkpoints/inception_v4.ckpt \
  --checkpoint_path_inception_resnet_v2=checkpoints/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_ens_adv_inception_resnet_v2=checkpoints/ens_adv_inception_resnet_v2_rename.ckpt \
  --checkpoint_path_resnet=checkpoints/resnet_v2_50.ckpt \
  --num_iter=8 \
  --momentum=0.0 \
  --use_existing=${USE_EXISTING} \
  --gpu=${GPU} \
  --batch_size=5 \
  --random_eps=1
