import os
os.environ["WANDB_PROJECT"]="dapo-proposal"

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import set_seed

set_seed(42)

from utils.math_dapo_reward import compute_score
from utils.metrics import *

SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer."

def math_dapo_reward(completions, **kwargs):
    return [
        compute_score(completion[0]["content"], ground_truth['ground_truth'])
        for completion, ground_truth in zip(completions, kwargs['reward_model'])
    ]    

def add_system_prompt(example):
    example['prompt'] = [{"role": "system", "content": SYSTEM_PROMPT}] + example['prompt']
    return example

train_dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train").shuffle(seed=42)
eval_dataset = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")

train_dataset = train_dataset.map(add_system_prompt)
eval_dataset = eval_dataset.map(add_system_prompt)

training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=2e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    fp16=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_generations=7,
    max_prompt_length=1024,
    num_train_epochs=1,
    max_grad_norm=0.2,
    output_dir="/output/Qwen2.5-Math-1.5B-Instruct-GRPO", 
    gradient_checkpointing=False,
    report_to='wandb', 
    run_name="GRPO-Qwen2.5-Math-1.5B-Instruct", 
    do_eval=True,
    eval_strategy="steps",
    top_p=1.0,
    temperature=1.0,
    max_completion_length=(1024 * 3)
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    reward_funcs=math_dapo_reward,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    exploration_metrics=[
        DiversityMetric(top_k=1000), 
        DistinctnessMetric(ngram=3), 
        SelfBleuMetric(ngram=3)
    ]
)
trainer.train()