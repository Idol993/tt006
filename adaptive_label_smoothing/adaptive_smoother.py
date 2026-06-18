import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from .beta_smoothing import BetaSmoothModule
from .dynamic_scheduler import DynamicSmoothingScheduler
from .topology_manager import TopologyManager
from .consistency_constraint import TopologyConsistencyLoss


class AdaptiveLabelSmoother(nn.Module):
    def __init__(self,
                 num_classes: int,
                 module_names: List[str],
                 base_smoothing: float = 0.1,
                 min_smoothing: float = 0.01,
                 max_smoothing: float = 0.5,
                 init_alpha: float = 2.0,
                 init_beta: float = 10.0,
                 learnable_params: bool = True,
                 consistency_weight: float = 0.1,
                 distance_type: str = "cosine",
                 merge_threshold: float = 0.05,
                 merge_window: int = 10,
                 min_group_size: int = 2,
                 max_groups: int = 8,
                 confidence_threshold: float = 0.8,
                 entropy_ratio_threshold: float = 0.5,
                 adjust_lr: float = 0.05,
                 overfitting_patience: int = 5,
                 overfitting_delta: float = 0.01,
                 eps: float = 1e-6):
        super().__init__()

        self.num_classes = num_classes
        self.module_names = list(module_names)
        self.eps = eps
        self._step_count = 0
        self._merge_split_interval = 50

        self.smoothing_modules = nn.ModuleDict()
        for name in self.module_names:
            self.smoothing_modules[name] = BetaSmoothModule(
                module_name=name,
                num_classes=num_classes,
                init_alpha=init_alpha,
                init_beta=init_beta,
                learnable_params=learnable_params,
                eps=eps
            )

        self.scheduler = DynamicSmoothingScheduler(
            num_classes=num_classes,
            base_smoothing=base_smoothing,
            min_smoothing=min_smoothing,
            max_smoothing=max_smoothing,
            confidence_threshold=confidence_threshold,
            entropy_ratio_threshold=entropy_ratio_threshold,
            adjust_lr=adjust_lr,
            overfitting_patience=overfitting_patience,
            overfitting_delta=overfitting_delta
        )

        self.topology_manager = TopologyManager(
            num_classes=num_classes,
            merge_threshold=merge_threshold,
            merge_window=merge_window,
            min_group_size=min_group_size,
            max_groups=max_groups,
            eps=eps
        )
        for name in self.module_names:
            self.topology_manager.register_module(
                module_name=name,
                init_alpha=init_alpha,
                init_beta=init_beta
            )

        self.consistency_loss = TopologyConsistencyLoss(
            num_classes=num_classes,
            consistency_weight=consistency_weight,
            distance_type=distance_type,
            eps=eps
        )

        self.register_buffer("_reference_labels", torch.zeros(0))

    def register_module(self, module_name: str,
                        init_alpha: Optional[float] = None,
                        init_beta: Optional[float] = None) -> None:
        if module_name in self.smoothing_modules:
            return

        alpha = init_alpha if init_alpha is not None else 2.0
        beta = init_beta if init_beta is not None else 10.0

        self.smoothing_modules[module_name] = BetaSmoothModule(
            module_name=module_name,
            num_classes=self.num_classes,
            init_alpha=alpha,
            init_beta=beta,
            learnable_params=True,
            eps=self.eps
        )

        self.topology_manager.register_module(
            module_name=module_name,
            init_alpha=alpha,
            init_beta=beta
        )

        self.module_names.append(module_name)

    def smooth_labels(self, module_name: str,
                      labels: torch.Tensor,
                      probs: torch.Tensor,
                      use_group_params: bool = True,
                      use_mean: bool = False) -> torch.Tensor:
        module = self.smoothing_modules[module_name]

        if use_group_params:
            alpha, beta = self.topology_manager.get_module_params(module_name)

            smoothing_mean = alpha / (alpha + beta)

            if use_mean:
                smoothing = smoothing_mean
            else:
                from torch.distributions import Beta
                beta_dist = Beta(alpha, beta)
                smoothing = beta_dist.sample((1,)).squeeze()

            if labels.dim() == 1:
                one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            else:
                one_hot = labels.float()

            uniform = torch.ones_like(one_hot) / self.num_classes
            smooth_labels = (1.0 - smoothing) * one_hot + smoothing * uniform

            return smooth_labels
        else:
            return module.smooth_labels(labels, probs, use_mean=use_mean)

    def smooth_all_modules(self, labels: torch.Tensor,
                           module_probs: Dict[str, torch.Tensor],
                           use_group_params: bool = True,
                           use_mean: bool = False) -> Dict[str, torch.Tensor]:
        smooth_labels_dict = {}

        for name in self.module_names:
            if name in module_probs:
                probs = module_probs[name]
            else:
                probs = torch.ones(labels.size(0), self.num_classes, device=labels.device) / self.num_classes

            smooth_labels = self.smooth_labels(
                module_name=name,
                labels=labels,
                probs=probs,
                use_group_params=use_group_params,
                use_mean=use_mean
            )
            smooth_labels_dict[name] = smooth_labels

        return smooth_labels_dict

    def update_module_stats(self, module_name: str, probs: torch.Tensor) -> None:
        module = self.smoothing_modules[module_name]
        module.update_stats(probs)

    def update_all_stats(self, module_probs: Dict[str, torch.Tensor]) -> None:
        for name in self.module_names:
            if name in module_probs:
                self.update_module_stats(name, module_probs[name])

    def adjust_smoothing(self, module_name: str) -> Dict[str, float]:
        module = self.smoothing_modules[module_name]

        current_mean, current_var, target_mean, target_var = self.scheduler.update_module_smoothing(module)

        self.topology_manager.record_smoothing_value(module_name, current_mean)

        group = self.topology_manager.get_group(module_name)
        if group is not None:
            target_alpha = self._beta_params_from_mean_var(target_mean, target_var)[0]
            target_beta = self._beta_params_from_mean_var(target_mean, target_var)[1]

            device = group.alpha.device
            target_alpha_t = torch.tensor(target_alpha, device=device, dtype=group.alpha.dtype)
            target_beta_t = torch.tensor(target_beta, device=device, dtype=group.beta.dtype)

            self.topology_manager.update_group_params(
                module_name=module_name,
                target_alpha=target_alpha_t,
                target_beta=target_beta_t,
                lr=self.scheduler.adjust_lr
            )

        return {
            "current_mean": current_mean,
            "current_var": current_var,
            "target_mean": target_mean,
            "target_var": target_var
        }

    def adjust_all_smoothing(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for name in self.module_names:
            results[name] = self.adjust_smoothing(name)
        return results

    def _beta_params_from_mean_var(self, mean: float, var: float) -> Tuple[float, float]:
        if var <= 0:
            var = 1e-5

        if mean <= 0 or mean >= 1:
            return 1.0, 1.0

        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)

        alpha = max(0.1, alpha)
        beta = max(0.1, beta)

        return alpha, beta

    def try_topology_reconstruction(self,
                                    module_val_accs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        self.topology_manager.clear_overfitting_marks()

        if module_val_accs is not None:
            for name, val_acc in module_val_accs.items():
                if self.scheduler.check_overfitting(name, val_acc):
                    self.topology_manager.mark_overfitting(name)

        split_groups = self.topology_manager.try_split_groups()

        merged_pairs = self.topology_manager.try_merge_groups()

        info = {
            "num_groups": self.topology_manager.num_groups(),
            "group_sizes": self.topology_manager.get_group_sizes(),
            "split_groups": split_groups,
            "merged_pairs": merged_pairs,
            "module_group_map": self.topology_manager.get_module_group_map()
        }

        return info

    def compute_consistency_loss(self,
                                 module_smooth_labels: Dict[str, torch.Tensor],
                                 reference_module: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.consistency_loss(module_smooth_labels, reference_module)

    def training_step(self,
                      labels: torch.Tensor,
                      module_probs: Dict[str, torch.Tensor],
                      module_val_accs: Optional[Dict[str, float]] = None,
                      use_group_params: bool = True) -> Dict[str, Any]:
        self._step_count += 1

        self.update_all_stats(module_probs)

        adjust_results = self.adjust_all_smoothing()

        smooth_labels_dict = self.smooth_all_modules(
            labels=labels,
            module_probs=module_probs,
            use_group_params=use_group_params
        )

        consistency_loss_val, consistency_scores = self.compute_consistency_loss(
            smooth_labels_dict
        )

        topology_info = None
        if self._step_count % self._merge_split_interval == 0:
            topology_info = self.try_topology_reconstruction(module_val_accs)

        result = {
            "smooth_labels": smooth_labels_dict,
            "consistency_loss": consistency_loss_val,
            "consistency_scores": consistency_scores,
            "adjust_results": adjust_results,
            "topology_info": topology_info,
            "step": self._step_count
        }

        return result

    def get_smoothing_info(self) -> Dict[str, Any]:
        info = {}

        for name in self.module_names:
            module = self.smoothing_modules[name]
            group = self.topology_manager.get_group(name)

            if group is not None:
                alpha, beta = group.alpha.item(), group.beta.item()
                group_mean = alpha / (alpha + beta)
                group_var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
            else:
                group_mean = None
                group_var = None

            info[name] = {
                "module_mean": module.smoothing_mean.item(),
                "module_var": module.smoothing_var.item(),
                "group_mean": group_mean,
                "group_var": group_var,
                "confidence": module.ema_confidence.item(),
                "marginal_entropy": module.ema_marginal_entropy.item(),
                "group_id": self.topology_manager.get_module_group_map().get(name)
            }

        info["num_groups"] = self.topology_manager.num_groups()
        info["group_sizes"] = self.topology_manager.get_group_sizes()

        return info

    def extra_repr(self) -> str:
        return (f"num_classes={self.num_classes}, "
                f"num_modules={len(self.module_names)}, "
                f"num_groups={self.topology_manager.num_groups()}")
