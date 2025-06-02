import datetime
import json
import os

# import matplotlib_inline.backend_inline
from functools import partial

import fire
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as hierarchy
import torch
import torch.nn.functional as F
from scipy.spatial.distance import squareform
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from peft import PeftConfig, PeftModel
from peft.tuners import lora
from peft.tuners.polar.layer import PoLARLinear

# matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")


def get_pairwise_dist_heatmap(W, save_path=None, title=None):
    # W = W.to(torch.float64)
    W_tilde = W / torch.norm(W, p=2, dim=1, keepdim=True)
    condensed_distances = torch.nn.functional.pdist(W_tilde.detach().cpu()).numpy()

    # get clustering. linkage matrix represents dendrogram
    linkage_matrix = hierarchy.linkage(condensed_distances, method="average")

    # leaves_order represents mapping between indices and leaves in dendrogram
    leaves_order = hierarchy.leaves_list(linkage_matrix)
    distances = squareform(condensed_distances)
    # reorder rows and columns according to clusters
    reordered_matrix = distances[leaves_order, :][:, leaves_order]

    plt.imshow(reordered_matrix, vmin=0.0, vmax=2.0)
    if title is not None:
        plt.title(title, wrap=True)
    plt.xlabel(r"$i$")
    plt.ylabel(r"$j$")
    plt.colorbar()
    plt.tight_layout()
    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight",
        )
    plt.clf()
    plt.close()


def read_file(file_path):
    with open(os.path.join(file_path)) as f:
        res = json.load(f)
    return res


def compute_stable_rank(delta_weight):
    sr = (torch.linalg.matrix_norm(delta_weight, ord="fro") ** 2) / (
        torch.linalg.matrix_norm(delta_weight, ord=2) ** 2
    )
    return sr.item()


def compute_model_stable_rank(model, cli_run_args, save_path):
    module_type = (
        lora.Linear
        if (cli_run_args["adapter"] == "lora" or cli_run_args["adapter"] == "dora")
        else PoLARLinear
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = []
    total_modules = sum(1 for m in model.modules() if isinstance(m, module_type))
    with tqdm(
        total=total_modules, desc="Computing Stable Ranks of Delta Weights"
    ) as pbar:
        for name, m in model.named_modules():
            if isinstance(m, module_type):
                current = cli_run_args.copy()
                registered_adapters = [k for k in m.lora_A.keys()]
                if len(registered_adapters) > 1:
                    raise ValueError(
                        "More than one adapter registered. Cannot uniquely determine get_delta_weight"
                    )
                else:
                    registered_adapter = registered_adapters[0]
                delta_weight = m.get_delta_weight(registered_adapter)
                if save_path is not None:
                    file_name = f"{name}_pairwise_dist.pdf"
                    save_path_img = os.path.join(save_path, file_name)
                    get_pairwise_dist_heatmap(
                        delta_weight,
                        save_path=save_path_img,
                        title=None,
                    )
                stable_rank_layer = compute_stable_rank(delta_weight)
                current["module"] = name
                current["stable_rank"] = stable_rank_layer
                # current["rank"] = torch.linalg.matrix_rank(delta_weight).item()
                df.append(current)
                pbar.update(1)
    df = pd.DataFrame(df)
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%dT%H%M%S%f")
    df.to_csv(os.path.join(save_path, f"{formatted_time}_stats_stable_ranks.csv"))
    return df


def main(peft_model_id, save_path):
    with open(os.path.join(peft_model_id, "cli_run_args.json")) as f:
        cli_run_args = json.load(f)
    adapter_name = cli_run_args["adapter"]
    parametrize_S = cli_run_args["parametrize_S"]
    print(f"Directory: {os.path.join(peft_model_id)}")
    save_path = os.path.join(peft_model_id, "stable_rank")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    print(peft_config)
    if adapter_name == "polar":
        custom_module_mapping = {
            nn.Linear: partial(PoLARLinear, parameterization_lora_S=parametrize_S)
        }
        peft_config._register_custom_module(custom_module_mapping)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path
    )
    if "stsb" in peft_model_id:
        # Replace the classification head for regression
        base_model.classifier = nn.Linear(
            in_features=base_model.classifier.in_features, out_features=1
        )

    model = PeftModel.from_pretrained(
        base_model,
        peft_model_id,
        config=peft_config,
        adapter_name=adapter_name,
    )
    df = compute_model_stable_rank(model, cli_run_args, save_path)
    return df


if __name__ == "__main__":
    fire.Fire(main)
