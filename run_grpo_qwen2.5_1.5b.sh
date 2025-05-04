#!/bin/bash

source activate trl

cd /workdir
ls

mkdir /output

wandb login bc8b7adfa1f1f1dab2bc7c8486496da696d0fd8a

CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model Qwen/Qwen2.5-Math-1.5B-Instruct --gpu_memory_utilization 0.85 --enable_prefix_caching true &

time sleep 60

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes 7 main.py