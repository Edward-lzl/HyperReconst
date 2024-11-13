#!/usr/bin/env bash 
#SBATCH --requeue

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

set -x
T=`date +%m%d-%H-%M`

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16 
export NCCL_MIN_NCHANNELS=16
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml 
export NCCL_IB_HCA=mlx5_0,mlx5_2 
export NCCL_IB_GID_INDEX=3 
export NCCL_IB_TIMEOUT=23 
export NCCL_IB_RETRY_CNT=7

PARTITION=$1
ACCOUNT=$2
JOB_NAME=$3 
# GPUS=${GPUS:-4} #-8/-4
# GPUS_PER_NODE=${GPUS_PER_NODE:-4} ###-4
# CPUS_PER_TASK=${CPUS_PER_TASK:-32} #-64/32
GPUS_PER_NODE=$4
GPUS=$5
CPUS_PER_TASK=$6
MODEL_CONFIG=$7
NUM_NODE=$8
PROJECT_NAME="${JOB_NAME}_${T}"
# random_port=$((1024 + RANDOM % 64512))

srun -p ${PARTITION} \
    -A ${ACCOUNT} \
    -N ${NUM_NODE} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --qos=gpugpu \

    accelerate launch --main_process_port=$MASTER_PORT --num_processes=$GPUS_PER_NODE main.py \
    --config $MODEL_CONFIG --mode train