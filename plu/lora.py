from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_dropout: float = 0.1


class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: LoraConfig):
        super().__init__()
        if config.r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.r = config.r
        self.scaling = config.lora_alpha / config.r
        self.dropout = nn.Dropout(config.lora_dropout)
        self.lora_A = nn.Linear(base_layer.in_features, config.r, bias=False)
        self.lora_B = nn.Linear(config.r, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        update = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return result + update.to(result.dtype)


def _matches_target(full_name: str, leaf_name: str, targets: Iterable[str]) -> bool:
    return any(leaf_name == target or full_name.endswith("." + target) for target in targets)


def freeze_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def apply_lora(model: nn.Module, config: LoraConfig) -> nn.Module:
    freeze_parameters(model)
    targets = tuple(config.target_modules)

    def replace(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _matches_target(full_name, child_name, targets):
                setattr(module, child_name, LoraLinear(child, config))
            else:
                replace(child, full_name)

    replace(model)
    return model


def iter_lora_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoraLinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def print_trainable_parameters(model: nn.Module) -> str:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    return f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}"


def save_lora_adapters(model: nn.Module, output_dir: str | Path, config: LoraConfig, base_model_name_or_path: str | None = None) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output_path / "adapter_model.bin")
    payload = {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(config.target_modules),
    }
    if base_model_name_or_path is not None:
        payload["base_model_name_or_path"] = base_model_name_or_path
    (output_path / "adapter_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
