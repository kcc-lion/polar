import json
import os
import sys
from functools import partial
from typing import List

import datasets
import fire
import lm_eval
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from lm_eval.loggers import WandbLogger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)

import wandb
from callbacks.landing_callback import LandingProjectionCallback
from callbacks.wandb_callback import OrthogonalityWandbCallback, StableRankWandbCallback
from lm_eval_prompter import generate_lm_eval_prompts
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.polar.layer import PoLARLinear
from prompter import Prompter
from stable_rank.analyze_sr import compute_model_stable_rank


def train(
    seed: int = 0,
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    dtype: str = "bf16",
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    # polar hyperparams
    parameterize_S: str = "identity",
    gradient_type: str = "landing",
    regularization_lambda: float = 1.0,
    init_lora_weights: str = "default",
    # llm hyperparams
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    # other
    save_strategy: str = "no",
    lr_scheduler_type: str = "cosine",
    do_eval: bool = True,
    add_stable_rank_callback: bool = False,
    activate_profiling: bool = False,
):
    if data_path == "metamathqa":
        eval_task = "gsm8k"
        num_fewshot = 0
        log_samples = True
    else:
        eval_task = data_path
        num_fewshot = None
        log_samples = False
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Fine-Tuning with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"dtype: {dtype}\n"
            f"adapter_name: {adapter_name}\n"
            f"lora_r: {lora_r}\n"
            f"regularization_lambda: {regularization_lambda}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"gradient_type: {gradient_type}\n"
            f"init_lora_weights: {init_lora_weights}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"eval_task: {eval_task}\n"
            f"num_fewshot: {num_fewshot}\n"
            f"init_lora_weights: {init_lora_weights}\n"
            f"save_strategy: {save_strategy}\n"
            f"lr_scheduler_type: {lr_scheduler_type}\n"
            f"do_eval: {do_eval}\n"
            f"add_stable_rank_callback: {add_stable_rank_callback}\n"
            f"activate_profiling: {activate_profiling}\n"
        )
        cli_run_args = {
            "epoch": num_epochs,
            "dataset": data_path,
            "model_name": base_model,
            "adapter": adapter_name,
            "lr": learning_rate,
            "seed": seed,
            "rank": lora_r,
            "dtype": dtype,
            "reg_lambda": regularization_lambda,
            "parametrize_S": parameterize_S,
            "gradient_type": gradient_type,
            "wandb_name": wandb_run_name,
            "eval_task": eval_task,
            "init_lora_weights": init_lora_weights,
            "save_strategy": save_strategy,
            "lr_scheduler_type": lr_scheduler_type,
            "do_eval": do_eval,
            "add_stable_rank_callback": add_stable_rank_callback,
            "activate_profiling": activate_profiling,
        }
        with open(os.path.join(output_dir, "cli_run_args.json"), "w") as file:
            json.dump(cli_run_args, file, indent=4)
    assert base_model, "Please specify a --base_model"
    assert dtype in ["bf16", "fp16"], f"Invalid dtype: {dtype}"
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device_map = "auto" if world_size > 1 else {"": 0}
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    transformers.set_seed(seed)
    if "llama" in base_model.lower():
        print(f"Using torch dtype: {torch_dtype}")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        gradient_checkpointing = False
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if base_model == "meta-llama/Llama-2-7b-hf":
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(f"Invalid base model: {base_model}")
        model = prepare_model_for_kbit_training(model)
    elif "gemma" in base_model.lower():
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=(
                torch.bfloat16 if "gemma-3" in base_model.lower() else torch.float16
            ),
            device_map=device_map,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if "gemma-3-27b" in base_model.lower():
            gradient_checkpointing = True
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        else:
            gradient_checkpointing = False
    else:
        raise ValueError(f"Invalid base model: {base_model}")

    def tokenize(prompt, add_eos_token=True):
        if "prompt" in prompt.keys():
            prompt = prompt["prompt"]
        else:
            prompt = prompter.generate_prompt(prompt, data_path=data_path)
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    if init_lora_weights == "default":
        init_strategy = True
    elif init_lora_weights == "random":
        init_strategy = False
    elif init_lora_weights == "symmetric_gaussian":
        init_strategy = "symmetric_gaussian"
    else:
        raise ValueError(f"Unknown init strategy for LoRA {init_lora_weights}")
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        # exclude vision tower in Gemma 3
        exclude_modules=(
            r"vision_tower.vision_model.encoder.layers.\S*"
            if "gemma-3" in base_model.lower()
            else None
        ),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        adapter_name=adapter_name,
        use_dora=True if adapter_name.lower() == "dora" else False,
        parameterize_S=parameterize_S,
        init_lora_weights=init_strategy,
    )
    if config.adapter_name.lower() == "polar":
        custom_module_mapping = {
            nn.Linear: partial(
                PoLARLinear, parameterization_lora_S=config.parameterize_S
            )
        }
        # register the new mapping
        config._register_custom_module(custom_module_mapping)
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        if data_path == "metamathqa":
            prompter = Prompter(template_name="metamathqa", verbose=False)
            data = load_dataset("meta-math/MetaMathQA")
        else:
            # dataset based on lm eval harness
            prompts = generate_lm_eval_prompts(task_name=data_path)
            data = datasets.Dataset.from_list(prompts)

    model.print_trainable_parameters()

    if val_set_size > 0:
        if isinstance(data, datasets.dataset_dict.DatasetDict):
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
        else:
            train_val = data.train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
        train_data = train_val["train"].shuffle().map(tokenize)
        val_data = train_val["test"].shuffle().map(tokenize)
    else:
        if isinstance(data, datasets.dataset_dict.DatasetDict):
            train_data = data["train"].shuffle().map(tokenize)
        else:
            train_data = data.shuffle().map(tokenize)
        val_data = None

    callbacks = []
    if add_stable_rank_callback:
        callbacks += [
            StableRankWandbCallback(layer_idxs=[1, 5, 10, 15], n_steps=10),
            OrthogonalityWandbCallback(rank=lora_r, n_steps=1),
        ]
    if adapter_name.lower() == "polar":
        callbacks += [
            LandingProjectionCallback(
                lambda_regul=regularization_lambda,
                gradient_type=gradient_type,
            )
        ]
    use_bf16 = "gemma" in base_model.lower()
    if "llama" in base_model.lower():
        use_bf16 = True if dtype == "bf16" else False
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type,
            # warmup_steps=100,
            warmup_ratio=0.03,
            gradient_checkpointing=gradient_checkpointing,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=0.02,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy=save_strategy,
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            ddp_find_unused_parameters=True if gradient_checkpointing else False,
            output_dir=output_dir,
            save_total_limit=1,
            seed=seed,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    if activate_profiling:
        from transformers.trainer_callback import TrainerCallback

        class ProfCallback(TrainerCallback):
            def __init__(self, prof):
                self.prof = prof

            def on_step_end(self, args, state, control, **kwargs):
                self.prof.step()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=5, wait=5, warmup=5, active=3, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(output_dir, "profiling")
            ),
            profile_memory=True,
            with_flops=False,
            with_stack=False,
            record_shapes=False,
        ) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train()
    else:
        trainer.train()

    model.save_pretrained(output_dir)
    if do_eval:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={base_model},peft={output_dir},trust_remote_code=True",
            tasks=eval_task,
            batch_size=micro_batch_size,
            num_fewshot=num_fewshot,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            log_samples=log_samples,
            write_out=True,
        )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            # save eval results to wandb and json
            wandb.run.config.update(
                {
                    "gradient_type": gradient_type,
                    "init_strategy": init_strategy,
                    "parametrize_S": parameterize_S,
                    "dataset": eval_task,
                    "lora_r": lora_r,
                }
            )
            wandb_logger = WandbLogger()
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            dumped = json.dumps(
                results,
                indent=2,
                default=lm_eval.utils.handle_non_serializable,
                ensure_ascii=False,
            )
            with open(
                os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8"
            ) as file:
                file.write(dumped)
            print(lm_eval.utils.make_table(results))
            wandb_logger.run.finish()

            # run stable rank
            df = compute_model_stable_rank(
                model,
                cli_run_args,
                os.path.join(output_dir, "stable_rank"),
            )
            print(f"Median Stable Rank: {df['stable_rank'].median()}")


if __name__ == "__main__":
    fire.Fire(train)
