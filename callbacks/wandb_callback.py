import warnings
from typing import Union

import torch
import torch.nn.functional as F
import transformers

import wandb
from peft.tuners import lora
from peft.tuners.polar.layer import PoLARLinear


class OrthogonalityWandbCallback(transformers.TrainerCallback):
    def __init__(self, n_steps=50, rank=8):
        self.n_steps = n_steps
        self.rank = rank

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.n_steps == 0:
            if state.is_local_process_zero and state.is_world_process_zero:
                orthogonality_A, orthogonality_B = self.compute_orthogonality(model)
                wandb.log(
                    {
                        "orthogonality_A": orthogonality_A,
                        "orthogonality_B": orthogonality_B,
                        "step": state.global_step,
                    }
                )

    @torch.no_grad()
    def compute_orthogonality(self, model):
        total_orthogonality_A, total_orthogonality_B = 0.0, 0.0
        lora_modules = [
            m
            for m in model.modules()
            if isinstance(m, PoLARLinear) or isinstance(m, lora.Linear)
        ]
        n_lora_modules = len(lora_modules)
        for m in lora_modules:
            lora_A, lora_B = m.lora_A["default"].weight, m.lora_B["default"].weight
            total_orthogonality_A += self.frobenius_norm_distance_to_identity(
                lora_A @ lora_A.t()
            )
            total_orthogonality_B += self.frobenius_norm_distance_to_identity(
                lora_B.t() @ lora_B
            )
        return (
            total_orthogonality_A / n_lora_modules,
            total_orthogonality_B / n_lora_modules,
        )

    def frobenius_norm_distance_to_identity(self, matrix):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square.")

        identity_matrix = torch.eye(
            matrix.shape[0], dtype=matrix.dtype, device=matrix.device
        )
        difference = matrix - identity_matrix
        frobenius_norm_distance = torch.norm(difference, p="fro") ** 2

        return frobenius_norm_distance.item()


class LandingProjectionCallback(transformers.TrainerCallback):
    def __init__(self, lambda_regul: float = 1.0, gradient_type: str = "landing"):
        self.lambda_regul = lambda_regul
        self.gradient_type = gradient_type

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.gradient_type == "landing":
            for param_name, param in model.named_parameters():
                if "lora_A" in param_name or "lora_B" in param_name:
                    landing_field = self.landing_step_opt(param, param.grad)
                    param.grad.data = landing_field
        elif self.gradient_type == "euclidean":
            for param_name, param in model.named_parameters():
                if "lora_A" in param_name or "lora_B" in param_name:
                    euclidean_field = self.euclidean_step(param, param.grad)
                    param.grad.data = euclidean_field
        else:
            raise NotImplementedError(f"Invalid gradient type {self.gradient_type}")

    def euclidean_step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            n, p = param.shape
            transposed = False
            # If the smaller dim is not the last dimension, transpose
            if p > n:
                param = param.transpose(-1, -2)
                grad = grad.transpose(-1, -2)
                n, p = param.shape
                transposed = True
            if not (p < n):
                raise ValueError(
                    "Expected p < n after re-orienting, but got n={}, p={}".format(n, p)
                )
            distance = torch.matmul(param.transpose(-1, -2), param) - torch.eye(
                p, device=param.device
            )  # (p, p)

            gradient_with_penalty = grad + self.lambda_regul * torch.matmul(
                param, distance
            )  # (n, p)
            # If we transposed at the beginning, transpose back
            if transposed:
                gradient_with_penalty = gradient_with_penalty.transpose(-1, -2)
            return gradient_with_penalty

    def landing_step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            n, p = param.shape
            transposed = False

            if p > n:
                param = param.transpose(-1, -2)
                grad = grad.transpose(-1, -2)
                n, p = param.shape
                transposed = True

            if not (p < n):
                raise ValueError(
                    "Expected p < n after re-orienting, but got n={}, p={}".format(n, p)
                )

            G = torch.matmul(grad, param.transpose(-1, -2))
            G = G - G.transpose(-1, -2)
            distance = torch.matmul(param.transpose(-1, -2), param) - torch.eye(
                p, device=param.device
            )

            landing = torch.matmul(G, param) + self.lambda_regul * torch.matmul(
                param, distance
            )
            if transposed:
                landing = landing.transpose(-1, -2)
            return landing

    def landing_step_opt(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            n, p = param.shape
            transposed = False

            if p > n:
                param = param.transpose(-1, -2)
                grad = grad.transpose(-1, -2)
                n, p = param.shape
                transposed = True
            landing = self.core_landing(param, grad)
            if transposed:
                landing = landing.transpose(-1, -2)
            return landing

    @torch.compile
    def core_landing(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        pt = param.transpose(-1, -2)
        G = torch.matmul(grad, pt)
        G = G - G.transpose(-1, -2)
        distance = torch.matmul(pt, param)
        distance.diagonal().add_(-1.0)
        landing = torch.matmul(G, param) + self.lambda_regul * torch.matmul(
            param, distance
        )
        return landing


class StableRankWandbCallback(transformers.TrainerCallback):
    def __init__(self, layer_idxs=[1, 5, 10, 15], n_steps=100):
        self.n_steps = n_steps
        self.layer_idxs = layer_idxs
        self.names = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.n_steps == 0 or state.global_step == 1:
            if state.is_local_process_zero and state.is_world_process_zero:
                log_dict = {"step": state.global_step}
                if self.names is None:
                    lora_module_names = [
                        n
                        for n, m in model.named_modules()
                        if (isinstance(m, PoLARLinear) or isinstance(m, lora.Linear))
                    ]
                    self.names = []
                    for n in lora_module_names:
                        for idx in self.layer_idxs:
                            if f"layers.{idx}." in n:
                                self.names.append(n)
                with torch.no_grad():
                    lora_modules = [
                        (n, m)
                        for n, m in model.named_modules()
                        if isinstance(m, PoLARLinear) or isinstance(m, lora.Linear)
                    ]
                    for n, m in lora_modules:
                        if n in self.names:
                            delta_weight = m.get_delta_weight(m.active_adapter[0])
                            stable_rank = self.compute_stable_rank(delta_weight)
                            log_dict[f"sr_{n}"] = stable_rank
                wandb.log(log_dict)

    def compute_stable_rank(self, delta_weight):
        s = torch.linalg.svdvals(delta_weight)
        s2 = s * s
        total_energy = s2.sum()
        stable_rank = (total_energy / s2.max()).item()
        return stable_rank
