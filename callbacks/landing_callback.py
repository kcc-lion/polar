import torch
import transformers


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
