import datetime
import json
import os
from functools import partial

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from peft import PeftConfig, PeftModel
from peft.tuners import lora
from peft.tuners.polar.layer import PoLARLinear


def compute_stable_rank(delta_weight):
    sr = (torch.linalg.matrix_norm(delta_weight, ord="fro") ** 2) / (
        torch.linalg.matrix_norm(delta_weight, ord=2) ** 2
    )
    return sr.item()


def get_stable_rank(peft_model_id):

    with open(os.path.join(peft_model_id, "cli_run_args.json")) as f:
        cli_run_args = json.load(f)

    adapter_name = cli_run_args["adapter"]
    parametrize_S = cli_run_args["parametrize_S"]
    print(f"Adapter: {adapter_name}, Parametrize S: {parametrize_S}")
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    if adapter_name == "polar":
        custom_module_mapping = {
            nn.Linear: partial(PoLARLinear, parameterization_lora_S=parametrize_S)
        }
        peft_config._register_custom_module(custom_module_mapping)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_id,
        config=peft_config,
        adapter_name=adapter_name,
    )
    rank = peft_config.r

    df = []
    module_type = (
        lora.Linear
        if (adapter_name == "lora") or (adapter_name == "dora")
        else PoLARLinear
    )
    total_modules = sum(1 for m in model.modules() if isinstance(m, module_type))
    with tqdm(total=total_modules, desc="Computing delta weights") as pbar:
        for name, m in model.named_modules():
            if isinstance(m, module_type):
                current = cli_run_args.copy()
                delta_weight = m.get_delta_weight(adapter_name)
                sr = compute_stable_rank(delta_weight)
                current["module"] = name
                current["stable_rank"] = sr
                print(f"\n {adapter_name},  {name}, sr: {sr:.2f}")
                pbar.update(1)
                df.append(current)
    return df


project_dir = "/path/to/dir"
df_all = []
dirs = os.listdir(project_dir)
dirs = [
    d
    for d in dirs
    if not d.startswith("slurm")
    and not d.startswith("wandb")
    and not d.startswith(".")
    and not d.endswith(".csv")
]
for peft_model_id in tqdm(dirs):
    print(f"Directory: {os.path.join(project_dir, peft_model_id)}")
    df_cur = get_stable_rank(os.path.join(project_dir, peft_model_id))
    df_all.extend(df_cur)
    print("\n")
    df = pd.DataFrame(df_all)
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
    df.to_csv(f"/path/to/outdir/{formatted_time}_rank_measures_commonsense.csv")
