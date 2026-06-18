import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import math

from .beta_smoothing import BetaSmoothModule


class DynamicSmoothingScheduler(nn.Module):
    def __init__(self,
                 num_classes: int,
                 base_smoothing: float = 0.1,
                 min_smoothing: float = 0.01,
                 max_smoothing: float = 0.5,
                 base_variance: float = 0.001,
                 min_variance: float = 1e-5,
                 max_variance: float = 0.01,
                 confidence_threshold: float = 0.8,
                 entropy_ratio_threshold: float = 0.5,
                 adjust_lr: float = 0.05,
                 overfitting_patience: int = 5,
                 overfitting_delta: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.base_smoothing = base_smoothing
        self.min_smoothing = min_smoothing
        self.max_smoothing = max_smoothing
        self.base_variance = base_variance
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.confidence_threshold = confidence_threshold
        self.entropy_ratio_threshold = entropy_ratio_threshold
        self.adjust_lr = adjust_lr

        self.overfitting_patience = overfitting_patience
        self.overfitting_delta = overfitting_delta

        self.max_entropy = math.log(num_classes)

        self._val_acc_history: Dict[str, List[float]] = {}
        self._best_val_acc: Dict[str, float] = {}
        self._overfit_counter: Dict[str, int] = {}

    def compute_target_smoothing(self, confidence: torch.Tensor,
                                 marginal_entropy: torch.Tensor) -> Tuple[float, float]:
        conf = confidence.item()
        ent = marginal_entropy.item()

        entropy_ratio = ent / self.max_entropy if self.max_entropy > 0 else 0.0

        if conf > self.confidence_threshold and entropy_ratio < self.entropy_ratio_threshold:
            target_mean = self.base_smoothing * 0.5
            target_var = self.base_variance * 0.5
        elif conf < 0.3 and entropy_ratio > 0.8:
            target_mean = min(self.base_smoothing * 1.5, self.max_smoothing)
            target_var = min(self.base_variance * 2.0, self.max_variance)
        else:
            confidence_factor = max(0.5, 1.0 - (conf - 0.5) * 0.8)
            entropy_factor = max(0.5, 1.0 - (entropy_ratio - 0.5) * 0.6)

            target_mean = self.base_smoothing * confidence_factor * entropy_factor
            target_var = self.base_variance * confidence_factor

        target_mean = max(self.min_smoothing, min(self.max_smoothing, target_mean))
        target_var = max(self.min_variance, min(self.max_variance, target_var))

        return target_mean, target_var

    def update_module_smoothing(self, module: BetaSmoothModule) -> Tuple[float, float, float, float]:
        confidence = module.ema_confidence
        marginal_entropy = module.ema_marginal_entropy

        target_mean, target_var = self.compute_target_smoothing(confidence, marginal_entropy)

        module.adjust_beta_params(target_mean, target_var, lr=self.adjust_lr)

        current_mean = module.smoothing_mean.item()
        current_var = module.smoothing_var.item()

        return current_mean, current_var, target_mean, target_var

    def check_overfitting(self, module_name: str, val_acc: float) -> bool:
        if module_name not in self._best_val_acc:
            self._best_val_acc[module_name] = val_acc
            self._overfit_counter[module_name] = 0
            self._val_acc_history[module_name] = [val_acc]
            return False

        self._val_acc_history[module_name].append(val_acc)

        if val_acc > self._best_val_acc[module_name] + self.overfitting_delta:
            self._best_val_acc[module_name] = val_acc
            self._overfit_counter[module_name] = 0
        elif val_acc < self._best_val_acc[module_name] - self.overfitting_delta:
            self._overfit_counter[module_name] += 1
        else:
            pass

        if self._overfit_counter[module_name] >= self.overfitting_patience:
            return True

        return False

    def reset_overfitting_tracker(self, module_name: str) -> None:
        if module_name in self._best_val_acc:
            self._best_val_acc[module_name] = 0.0
            self._overfit_counter[module_name] = 0
            self._val_acc_history[module_name] = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        state[prefix + '_val_acc_history'] = self._val_acc_history
        state[prefix + '_best_val_acc'] = self._best_val_acc
        state[prefix + '_overfit_counter'] = self._overfit_counter
        return state

    def load_state_dict(self, state_dict, strict=True):
        keys_to_extract = ['_val_acc_history', '_best_val_acc', '_overfit_counter']
        extracted = {}
        filtered_state = {}
        for k, v in state_dict.items():
            base_key = k.split('.')[-1] if '.' in k else k
            if base_key in keys_to_extract:
                extracted[base_key] = v
            else:
                filtered_state[k] = v

        super().load_state_dict(filtered_state, strict=strict)

        if '_val_acc_history' in extracted:
            self._val_acc_history = extracted['_val_acc_history']
        if '_best_val_acc' in extracted:
            self._best_val_acc = extracted['_best_val_acc']
        if '_overfit_counter' in extracted:
            self._overfit_counter = extracted['_overfit_counter']

        return {}
