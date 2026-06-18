import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set, Literal
from dataclasses import dataclass, field
import json
import csv
import io
import hashlib

from .beta_smoothing import BetaSmoothModule
from .dynamic_scheduler import DynamicSmoothingScheduler
from .topology_manager import TopologyManager
from .consistency_constraint import TopologyConsistencyLoss


CHECKPOINT_VERSION = "4.1.0"


@dataclass
class CheckpointLoadReport:
    loaded_modules: List[str] = field(default_factory=list)
    skipped_modules: List[str] = field(default_factory=list)
    missing_modules: List[str] = field(default_factory=list)
    extra_modules: List[str] = field(default_factory=list)
    version_match: bool = True
    checkpoint_version: str = ""
    current_version: str = CHECKPOINT_VERSION
    hyperparams_match: bool = True
    hyperparams_mismatch_keys: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"CheckpointLoadReport:"]
        lines.append(f"  version: ckpt={self.checkpoint_version} "
                     f"current={self.current_version} match={self.version_match}")
        lines.append(f"  hyperparams_match={self.hyperparams_match} "
                     f"mismatch_keys={self.hyperparams_mismatch_keys}")
        lines.append(f"  loaded ({len(self.loaded_modules)}): {self.loaded_modules}")
        lines.append(f"  skipped ({len(self.skipped_modules)}): {self.skipped_modules}")
        lines.append(f"  missing_in_smoother ({len(self.missing_modules)}): {self.missing_modules}")
        lines.append(f"  extra_in_ckpt ({len(self.extra_modules)}): {self.extra_modules}")
        if self.messages:
            lines.append(f"  messages ({len(self.messages)}):")
            for m in self.messages:
                lines.append(f"    - {m}")
        return "\n".join(lines)


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
                 warmup_steps: int = 0,
                 eps: float = 1e-6):
        super().__init__()

        self.num_classes = num_classes
        self.module_names = list(module_names)
        self.eps = eps
        self._step_count = 0
        self._merge_split_interval = 50
        self._warmup_steps = warmup_steps
        self._frozen_modules: Set[str] = set()

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

        self._diagnostic_log: List[Dict[str, Any]] = []
        self._enable_diagnostics: bool = True

        self._hyperparams = {
            "num_classes": num_classes,
            "base_smoothing": base_smoothing,
            "min_smoothing": min_smoothing,
            "max_smoothing": max_smoothing,
            "init_alpha": init_alpha,
            "init_beta": init_beta,
            "learnable_params": learnable_params,
            "consistency_weight": consistency_weight,
            "distance_type": distance_type,
            "merge_threshold": merge_threshold,
            "merge_window": merge_window,
            "min_group_size": min_group_size,
            "max_groups": max_groups,
            "confidence_threshold": confidence_threshold,
            "entropy_ratio_threshold": entropy_ratio_threshold,
            "adjust_lr": adjust_lr,
            "overfitting_patience": overfitting_patience,
            "overfitting_delta": overfitting_delta,
            "warmup_steps": warmup_steps,
            "eps": eps,
        }

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

        effective_use_group = use_group_params and not module.frozen

        if effective_use_group:
            alpha_orig, beta_orig = self.topology_manager.get_module_params(module_name)
            target_device = labels.device
            alpha = alpha_orig.to(target_device) if alpha_orig.device != target_device else alpha_orig
            beta = beta_orig.to(target_device) if beta_orig.device != target_device else beta_orig

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
            dtype = group.alpha.dtype
            target_alpha_t = torch.tensor(target_alpha, device=device, dtype=dtype)
            target_beta_t = torch.tensor(target_beta, device=device, dtype=dtype)

            self.topology_manager.update_group_params(
                module_name=module_name,
                target_alpha=target_alpha_t,
                target_beta=target_beta_t,
                lr=self.scheduler.adjust_lr,
                frozen_modules=self._frozen_modules,
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
            if self.is_frozen(name):
                continue
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

        frozen_list = self.get_frozen_modules()

        split_groups, split_details = self.topology_manager.try_split_groups(skip_modules=frozen_list)

        merged_pairs, merged_details = self.topology_manager.try_merge_groups(skip_modules=frozen_list)

        info = {
            "num_groups": self.topology_manager.num_groups(),
            "group_sizes": self.topology_manager.get_group_sizes(),
            "split_groups": split_groups,
            "merged_pairs": merged_pairs,
            "split_details": split_details,
            "merged_details": merged_details,
            "module_group_map": self.topology_manager.get_module_group_map()
        }

        return info

    def compute_consistency_loss(self,
                                 module_smooth_labels: Dict[str, torch.Tensor],
                                 reference_module: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.consistency_loss(module_smooth_labels, reference_module)

    @property
    def in_warmup(self) -> bool:
        return self._step_count <= self._warmup_steps

    @property
    def warmup_steps(self) -> int:
        return self._warmup_steps

    def set_warmup_steps(self, steps: int) -> None:
        self._warmup_steps = max(0, steps)

    def freeze_module(self, module_name: str) -> None:
        if module_name in self.smoothing_modules:
            self._frozen_modules.add(module_name)
            self.smoothing_modules[module_name].freeze()

    def unfreeze_module(self, module_name: str) -> None:
        if module_name in self._frozen_modules:
            self._frozen_modules.discard(module_name)
            self.smoothing_modules[module_name].unfreeze()

    def freeze_all(self) -> None:
        for name in self.module_names:
            self.freeze_module(name)

    def unfreeze_all(self) -> None:
        for name in self.module_names:
            self.unfreeze_module(name)

    def is_frozen(self, module_name: str) -> bool:
        return module_name in self._frozen_modules

    def get_frozen_modules(self) -> List[str]:
        return sorted(self._frozen_modules)

    def training_loss(self,
                      module_logits: Dict[str, torch.Tensor],
                      labels: torch.Tensor,
                      module_val_accs: Optional[Dict[str, float]] = None,
                      use_group_params: bool = True,
                      reduction: str = "mean",
                      return_details: bool = False,
                      loss_weights: Optional[Dict[str, float]] = None,
                      ignore_modules: Optional[List[str]] = None,
                      consistency_modules: Optional[List[str]] = None) -> Any:
        loss_result = self.compute_losses(
            module_logits=module_logits,
            labels=labels,
            module_val_accs=module_val_accs,
            use_group_params=use_group_params,
            reduction=reduction,
            loss_weights=loss_weights,
            ignore_modules=ignore_modules,
            consistency_modules=consistency_modules,
        )

        total_loss = loss_result["total_loss"]

        target_device = labels.device
        if total_loss.device != target_device:
            total_loss = total_loss.to(target_device)

        if not return_details:
            return total_loss

        smooth_labels = loss_result["smooth_labels"]
        for name, sl in smooth_labels.items():
            if sl.device != target_device:
                smooth_labels[name] = sl.to(target_device)

        cons_loss = loss_result["consistency_loss"]
        if isinstance(cons_loss, torch.Tensor) and cons_loss.device != target_device:
            cons_loss = cons_loss.to(target_device)

        cls_loss = loss_result["classification_loss"]
        if isinstance(cls_loss, torch.Tensor) and cls_loss.device != target_device:
            cls_loss = cls_loss.to(target_device)

        per_module_loss = loss_result["per_module_loss"]
        for name, pl in per_module_loss.items():
            if isinstance(pl, torch.Tensor) and pl.device != target_device:
                per_module_loss[name] = pl.to(target_device)

        log = dict(loss_result.get("flat_log", {}))

        details = {
            "total_loss": total_loss,
            "classification_loss": cls_loss,
            "consistency_loss": cons_loss,
            "per_module_loss": per_module_loss,
            "smooth_labels": smooth_labels,
            "smoothing_info": loss_result["smoothing_info"],
            "topology_info": loss_result["topology_info"],
            "step": loss_result["step"],
            "log": log,
            "flat_log": log,
            "active_modules": loss_result.get("active_modules", list(self.module_names)),
            "consistency_modules": loss_result.get("consistency_modules", list(self.module_names)),
            "ignored_modules": loss_result.get("ignored_modules", []),
        }

        return total_loss, details

    def training_step(self,
                      labels: torch.Tensor,
                      module_probs: Dict[str, torch.Tensor],
                      module_val_accs: Optional[Dict[str, float]] = None,
                      use_group_params: bool = True) -> Dict[str, Any]:
        self._step_count += 1

        self.update_all_stats(module_probs)

        if self.in_warmup:
            adjust_results = {}
        else:
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
        if not self.in_warmup and self._step_count % self._merge_split_interval == 0:
            if self.get_rank() == 0:
                topology_info = self.try_topology_reconstruction(module_val_accs)
            if self.is_distributed() and self.get_world_size() > 1:
                topology_info = self._dist_broadcast_topology_decision(topology_info)

        self._record_diagnostic(self._step_count, adjust_results, topology_info)

        global_stats = None
        if self.is_distributed() and self.get_world_size() > 1:
            global_stats = self.aggregate_module_stats_distributed()

        result = {
            "smooth_labels": smooth_labels_dict,
            "consistency_loss": consistency_loss_val,
            "consistency_scores": consistency_scores,
            "adjust_results": adjust_results,
            "topology_info": topology_info,
            "step": self._step_count,
            "in_warmup": self.in_warmup,
            "dist_rank": self.get_rank(),
            "dist_world_size": self.get_world_size(),
            "global_module_stats": global_stats,
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

    def compute_losses(self,
                       module_logits: Dict[str, torch.Tensor],
                       labels: torch.Tensor,
                       module_val_accs: Optional[Dict[str, float]] = None,
                       use_group_params: bool = True,
                       reduction: str = "mean",
                       loss_weights: Optional[Dict[str, float]] = None,
                       ignore_modules: Optional[List[str]] = None,
                       consistency_modules: Optional[List[str]] = None) -> Dict[str, Any]:
        ignore_set = set(ignore_modules) if ignore_modules else set()
        cons_set = set(consistency_modules) if consistency_modules else set(self.module_names)
        cons_set = cons_set - ignore_set

        active_module_names = [n for n in self.module_names if n not in ignore_set]

        module_probs = {}
        for name, logits in module_logits.items():
            module_probs[name] = F.softmax(logits, dim=-1)

        train_result = self.training_step(
            labels=labels,
            module_probs=module_probs,
            module_val_accs=module_val_accs,
            use_group_params=use_group_params
        )

        smooth_labels_dict = train_result["smooth_labels"]

        if consistency_modules is not None or ignore_modules is not None:
            cons_labels = {n: smooth_labels_dict[n] for n in smooth_labels_dict if n in cons_set}
            if len(cons_labels) >= 2:
                consistency_loss_val, consistency_scores = self.compute_consistency_loss(cons_labels)
            else:
                consistency_loss_val = torch.tensor(0.0, device=labels.device)
                consistency_scores = {}
        else:
            consistency_loss_val = train_result["consistency_loss"]
            consistency_scores = train_result["consistency_scores"]

        per_module_loss = {}
        total_cls_loss = torch.tensor(0.0, device=labels.device)
        total_weight = 0.0

        for name in active_module_names:
            if name not in module_logits:
                continue

            logits = module_logits[name]
            smooth_labels = smooth_labels_dict[name]

            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smooth_labels * log_probs).sum(dim=-1)

            if reduction == "mean":
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()

            weight = 1.0
            if loss_weights and name in loss_weights:
                weight = float(loss_weights[name])

            weighted_loss = loss * weight
            per_module_loss[name] = weighted_loss
            total_cls_loss = total_cls_loss + weighted_loss
            total_weight += weight

        if reduction == "mean" and len(per_module_loss) > 0 and total_weight > 0:
            avg_cls_loss = total_cls_loss / total_weight * len(per_module_loss)
        else:
            avg_cls_loss = total_cls_loss

        total_loss = avg_cls_loss + consistency_loss_val

        smoothing_info = self.get_smoothing_info()

        flat_log = self._build_flat_log(
            step=train_result["step"],
            total_loss=total_loss,
            classification_loss=avg_cls_loss,
            consistency_loss=consistency_loss_val,
            per_module_loss=per_module_loss,
            smoothing_info=smoothing_info,
            topology_info=train_result["topology_info"],
            in_warmup=train_result.get("in_warmup", False),
        )

        if self.is_distributed() and self.get_world_size() > 1:
            flat_log = self.build_distributed_flat_log(flat_log)
        else:
            enhanced = {"local/" + k: v for k, v in flat_log.items()}
            enhanced.update({"global/" + k: v for k, v in flat_log.items()})
            enhanced.update(flat_log)
            enhanced["global/dist_rank"] = 0
            enhanced["global/dist_world_size"] = 1
            flat_log = enhanced

        result = {
            "total_loss": total_loss,
            "classification_loss": avg_cls_loss,
            "consistency_loss": consistency_loss_val,
            "per_module_loss": per_module_loss,
            "smooth_labels": smooth_labels_dict,
            "consistency_scores": consistency_scores,
            "adjust_results": train_result["adjust_results"],
            "topology_info": train_result["topology_info"],
            "step": train_result["step"],
            "smoothing_info": smoothing_info,
            "flat_log": flat_log,
            "in_warmup": train_result.get("in_warmup", False),
            "active_modules": active_module_names,
            "consistency_modules": list(cons_set),
            "ignored_modules": list(ignore_set),
        }

        return result

    def _build_flat_log(self,
                        step: int,
                        total_loss: torch.Tensor,
                        classification_loss: torch.Tensor,
                        consistency_loss: torch.Tensor,
                        per_module_loss: Dict[str, torch.Tensor],
                        smoothing_info: Dict[str, Any],
                        topology_info: Optional[Dict[str, Any]],
                        in_warmup: bool) -> Dict[str, Any]:
        def _v(x):
            if isinstance(x, torch.Tensor):
                return float(x.item())
            return float(x) if x is not None else 0.0

        log: Dict[str, Any] = {}
        log["step"] = step
        log["loss/total"] = _v(total_loss)
        log["loss/classification"] = _v(classification_loss)
        log["loss/consistency"] = _v(consistency_loss)
        log["train/in_warmup"] = int(in_warmup)
        log["train/frozen_count"] = len(self.get_frozen_modules())
        log["topology/num_groups"] = smoothing_info.get("num_groups", 0)

        for name, loss_t in per_module_loss.items():
            log[f"loss/cls/{name}"] = _v(loss_t)

        for name in self.module_names:
            mi = smoothing_info.get(name, {})
            log[f"smooth/mean/{name}"] = mi.get("module_mean", 0.0)
            log[f"smooth/var/{name}"] = mi.get("module_var", 0.0)
            log[f"stats/confidence/{name}"] = mi.get("confidence", 0.0)
            log[f"stats/entropy/{name}"] = mi.get("marginal_entropy", 0.0)
            gid = mi.get("group_id", -1)
            log[f"topology/group_id/{name}"] = gid if gid is not None else -1

        if topology_info is not None:
            merged = topology_info.get("merged_pairs", [])
            split = topology_info.get("split_groups", [])
            log["topology/merged_count"] = len(merged)
            log["topology/split_count"] = len(split)
        else:
            log["topology/merged_count"] = 0
            log["topology/split_count"] = 0

        return log

    @staticmethod
    def _module_list_checksum(module_names: List[str]) -> str:
        h = hashlib.md5()
        for n in sorted(module_names):
            h.update(n.encode("utf-8"))
        return h.hexdigest()[:12]

    def _build_diagnostic_summary(self) -> Dict[str, Any]:
        try:
            summary = self.get_module_summary()
            compact: Dict[str, Any] = {}
            for name, s in summary.items():
                compact[name] = {
                    "group_id": s.get("current_group_id"),
                    "avg_smoothing_mean": round(s.get("avg_smoothing_mean", 0), 6),
                    "max_smoothing_var": round(s.get("max_smoothing_var", 0), 8),
                    "merge_count": s.get("merge_count", 0),
                    "split_count": s.get("split_count", 0),
                    "num_samples": s.get("num_samples", 0),
                }
            return {
                "total_steps": self._step_count,
                "num_groups": self.topology_manager.num_groups(),
                "module_stats": compact,
            }
        except Exception:
            return {"total_steps": self._step_count, "num_groups": self.topology_manager.num_groups()}

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        state[prefix + 'meta_version'] = CHECKPOINT_VERSION
        state[prefix + 'meta_hyperparams'] = dict(self._hyperparams)
        state[prefix + 'meta_module_names'] = list(self.module_names)
        state[prefix + 'meta_module_names_checksum'] = self._module_list_checksum(self.module_names)
        state[prefix + 'meta_diagnostic_summary'] = self._build_diagnostic_summary()

        state[prefix + 'module_names'] = list(self.module_names)
        state[prefix + 'num_classes'] = self.num_classes
        state[prefix + 'step_count'] = self._step_count
        state[prefix + 'merge_split_interval'] = self._merge_split_interval
        state[prefix + 'enable_diagnostics'] = self._enable_diagnostics
        state[prefix + 'diagnostic_log'] = list(self._diagnostic_log)

        state[prefix + 'scheduler_val_acc_history'] = dict(self.scheduler._val_acc_history)
        state[prefix + 'scheduler_best_val_acc'] = dict(self.scheduler._best_val_acc)
        state[prefix + 'scheduler_overfit_counter'] = dict(self.scheduler._overfit_counter)

        state[prefix + 'warmup_steps'] = self._warmup_steps
        state[prefix + 'frozen_modules'] = sorted(self._frozen_modules)

        topo_state = {}
        for gid, group in self.topology_manager._groups.items():
            topo_state[gid] = {
                'group_id': group.group_id,
                'module_names': list(group.module_names),
                'alpha': group.alpha.data.clone(),
                'beta': group.beta.data.clone(),
                'merge_count': group.merge_count,
                'split_count': group.split_count,
            }
        state[prefix + 'topology_groups'] = topo_state
        state[prefix + 'topology_module_to_group'] = dict(self.topology_manager._module_to_group)
        state[prefix + 'topology_module_history'] = {k: list(v) for k, v in self.topology_manager._module_history.items()}
        state[prefix + 'topology_next_group_id'] = self.topology_manager._next_group_id
        state[prefix + 'topology_step_count'] = self.topology_manager._step_count
        state[prefix + 'topology_overfitting_modules'] = list(self.topology_manager._overfitting_modules)

        return state

    def load_state_dict(self, state_dict, strict: bool = True,
                        load_strategy: Literal["strict", "intersection", "mapping"] = "strict",
                        module_mapping: Optional[Dict[str, str]] = None,
                        return_report: bool = False):
        LoadStrategyT = Literal["strict", "intersection", "mapping"]
        own_prefix_keys = [
            'module_names', 'num_classes', 'step_count',
            'merge_split_interval', 'enable_diagnostics', 'diagnostic_log',
            'scheduler_val_acc_history', 'scheduler_best_val_acc', 'scheduler_overfit_counter',
            'topology_groups', 'topology_module_to_group', 'topology_module_history',
            'topology_next_group_id', 'topology_step_count', 'topology_overfitting_modules',
            'warmup_steps', 'frozen_modules',
            'meta_version', 'meta_hyperparams', 'meta_module_names',
            'meta_module_names_checksum', 'meta_diagnostic_summary',
        ]

        report = CheckpointLoadReport()
        report.current_version = CHECKPOINT_VERSION
        module_mapping = module_mapping or {}

        extracted: Dict[str, Any] = {}
        filtered_state = {}
        prefix_found = ""

        for k, v in state_dict.items():
            is_own = False
            for ok in own_prefix_keys:
                if k.endswith('.' + ok):
                    extracted[ok] = v
                    is_own = True
                    if not prefix_found:
                        prefix_found = k[:-(len(ok) + 1)]
                    break
                elif k == ok:
                    extracted[ok] = v
                    is_own = True
                    break
            if not is_own:
                filtered_state[k] = v

        ckpt_module_names: List[str] = list(extracted.get('meta_module_names',
                                                          extracted.get('module_names', [])))
        report.checkpoint_version = str(extracted.get('meta_version', ''))
        report.version_match = (report.checkpoint_version == CHECKPOINT_VERSION)
        if not report.version_match and ckpt_module_names:
            report.messages.append(
                f"Checkpoint version mismatch: ckpt={report.checkpoint_version} "
                f"current={CHECKPOINT_VERSION}; loading best-effort"
            )

        ckpt_hp = extracted.get('meta_hyperparams', {})
        if ckpt_hp:
            mismatch_keys = []
            for k, v in self._hyperparams.items():
                if k in ckpt_hp and ckpt_hp[k] != v:
                    mismatch_keys.append(k)
            report.hyperparams_mismatch_keys = mismatch_keys
            report.hyperparams_match = len(mismatch_keys) == 0
            if mismatch_keys:
                report.messages.append(f"Hyperparams mismatch keys: {mismatch_keys}")

        ckpt_module_set = set(ckpt_module_names)
        current_module_set = set(self.module_names)

        if load_strategy == "strict":
            if ckpt_module_names and sorted(ckpt_module_names) != sorted(self.module_names):
                missing = sorted(current_module_set - ckpt_module_set)
                extra = sorted(ckpt_module_set - current_module_set)
                msg = f"Module list mismatch in strict mode. missing={missing}, extra={extra}"
                if return_report:
                    report.messages.append(msg)
                    report.missing_modules = missing
                    report.extra_modules = extra
                    return report
                else:
                    raise RuntimeError(msg)

            if not report.version_match:
                msg = (f"Checkpoint version {report.checkpoint_version} != current "
                       f"{CHECKPOINT_VERSION} in strict mode")
                if return_report:
                    report.messages.append(msg)
                    return report
                else:
                    raise RuntimeError(msg)

            name_remap = {n: n for n in self.module_names}
            for n in self.module_names:
                report.loaded_modules.append(n)

        elif load_strategy == "intersection":
            common = sorted(current_module_set & ckpt_module_set)
            only_current = sorted(current_module_set - ckpt_module_set)
            only_ckpt = sorted(ckpt_module_set - current_module_set)
            name_remap = {n: n for n in common}
            report.loaded_modules = common
            report.skipped_modules = only_current
            report.missing_modules = only_current
            report.extra_modules = only_ckpt
            if only_current:
                report.messages.append(f"New modules not in ckpt, skipped loading: {only_current}")
            if only_ckpt:
                report.messages.append(f"Ckpt modules not in current model: {only_ckpt}")

        elif load_strategy == "mapping":
            name_remap: Dict[str, str] = {}
            inv_map = {v: k for k, v in module_mapping.items()}
            for current_name in self.module_names:
                ckpt_name = module_mapping.get(current_name, current_name)
                if ckpt_name in ckpt_module_set:
                    name_remap[current_name] = ckpt_name
                    report.loaded_modules.append(current_name)
                else:
                    report.skipped_modules.append(current_name)
                    report.missing_modules.append(current_name)
                    report.messages.append(f"No mapping / no ckpt entry for module {current_name}")
            for ckpt_name in ckpt_module_names:
                matched = False
                for cn in self.module_names:
                    if name_remap.get(cn) == ckpt_name:
                        matched = True
                        break
                if not matched and ckpt_name not in current_module_set:
                    report.extra_modules.append(ckpt_name)
        else:
            raise ValueError(f"Unknown load_strategy={load_strategy}")

        if strict:
            super().load_state_dict(filtered_state, strict=strict)
        else:
            try:
                super().load_state_dict(filtered_state, strict=False)
            except Exception as e:
                report.messages.append(f"super().load_state_dict warning: {e}")

        ckpt_topology_groups = extracted.get('topology_groups', {})
        from .topology_manager import SmoothGroup

        remapped_groups = {}
        for gid, gdata in ckpt_topology_groups.items():
            remapped_names = []
            for m_name in gdata.get('module_names', []):
                matched_current = None
                for cn, ck in name_remap.items():
                    if ck == m_name:
                        matched_current = cn
                        break
                if matched_current is not None and matched_current in report.loaded_modules:
                    remapped_names.append(matched_current)
            if remapped_names:
                remapped_groups[gid] = {
                    **gdata,
                    'module_names': remapped_names,
                }

        if 'module_names' in extracted:
            final_module_names = []
            for cn in self.module_names:
                if cn in report.loaded_modules:
                    final_module_names.append(cn)
                else:
                    final_module_names.append(cn)
            self.module_names = final_module_names

        if 'step_count' in extracted:
            self._step_count = extracted['step_count']
        if 'merge_split_interval' in extracted:
            self._merge_split_interval = extracted['merge_split_interval']
        if 'enable_diagnostics' in extracted:
            self._enable_diagnostics = extracted['enable_diagnostics']
        if 'diagnostic_log' in extracted:
            orig_log = list(extracted['diagnostic_log'])
            if load_strategy == "mapping" and module_mapping:
                remapped_log = []
                for entry in orig_log:
                    new_entry = dict(entry)
                    if "module_stats" in entry:
                        new_ms = {}
                        for stat_name, stat_val in entry["module_stats"].items():
                            matched_current = None
                            for cn, ck in name_remap.items():
                                if ck == stat_name:
                                    matched_current = cn
                                    break
                            if matched_current is not None:
                                new_ms[matched_current] = dict(stat_val)
                        new_entry["module_stats"] = new_ms
                    remapped_log.append(new_entry)
                self._diagnostic_log = remapped_log
            else:
                self._diagnostic_log = orig_log

        if 'scheduler_val_acc_history' in extracted:
            orig = dict(extracted['scheduler_val_acc_history'])
            remapped = {}
            for m_name, v in orig.items():
                matched_current = None
                for cn, ck in name_remap.items():
                    if ck == m_name:
                        matched_current = cn
                        break
                if matched_current is not None:
                    remapped[matched_current] = v
            self.scheduler._val_acc_history = remapped
        if 'scheduler_best_val_acc' in extracted:
            orig = dict(extracted['scheduler_best_val_acc'])
            remapped = {}
            for m_name, v in orig.items():
                matched_current = None
                for cn, ck in name_remap.items():
                    if ck == m_name:
                        matched_current = cn
                        break
                if matched_current is not None:
                    remapped[matched_current] = v
            self.scheduler._best_val_acc = remapped
        if 'scheduler_overfit_counter' in extracted:
            orig = dict(extracted['scheduler_overfit_counter'])
            remapped = {}
            for m_name, v in orig.items():
                matched_current = None
                for cn, ck in name_remap.items():
                    if ck == m_name:
                        matched_current = cn
                        break
                if matched_current is not None:
                    remapped[matched_current] = v
            self.scheduler._overfit_counter = remapped

        if 'warmup_steps' in extracted:
            self._warmup_steps = extracted['warmup_steps']
        if 'frozen_modules' in extracted:
            ckpt_frozen = list(extracted['frozen_modules'])
            remapped_frozen = set()
            for m_name in ckpt_frozen:
                matched_current = None
                for cn, ck in name_remap.items():
                    if ck == m_name:
                        matched_current = cn
                        break
                if matched_current is not None:
                    remapped_frozen.add(matched_current)
            self._frozen_modules = remapped_frozen
            for name in self.module_names:
                if name in self.smoothing_modules:
                    mod = self.smoothing_modules[name]
                    if name in self._frozen_modules:
                        mod.freeze()
                    else:
                        mod.unfreeze()

        new_groups: Dict[int, SmoothGroup] = {}
        for gid, gdata in remapped_groups.items():
            group = SmoothGroup(
                group_id=gdata['group_id'],
                module_names=list(gdata['module_names']),
                alpha=gdata['alpha'].clone(),
                beta=gdata['beta'].clone(),
                merge_count=gdata.get('merge_count', 0),
                split_count=gdata.get('split_count', 0),
            )
            new_groups[gid] = group
        if new_groups:
            self.topology_manager._groups = new_groups

        if 'topology_module_to_group' in extracted:
            orig = dict(extracted['topology_module_to_group'])
            remapped = {}
            for m_name, gid in orig.items():
                matched_current = None
                for cn, ck in name_remap.items():
                    if ck == m_name:
                        matched_current = cn
                        break
                if matched_current is not None:
                    remapped[matched_current] = gid
            self.topology_manager._module_to_group = remapped
        if 'topology_module_history' in extracted:
            orig = {k: list(v) for k, v in extracted['topology_module_history'].items()}
            remapped = {}
            for m_name, v in orig.items():
                matched_current = None
                for cn, ck in name_remap.items():
                    if ck == m_name:
                        matched_current = cn
                        break
                if matched_current is not None:
                    remapped[matched_current] = list(v)
            self.topology_manager._module_history = remapped
        if 'topology_next_group_id' in extracted:
            self.topology_manager._next_group_id = extracted['topology_next_group_id']
        if 'topology_step_count' in extracted:
            self.topology_manager._step_count = extracted['topology_step_count']
        if 'topology_overfitting_modules' in extracted:
            orig = list(extracted['topology_overfitting_modules'])
            remapped = set()
            for m_name in orig:
                matched_current = None
                for cn, ck in name_remap.items():
                    if ck == m_name:
                        matched_current = cn
                        break
                if matched_current is not None:
                    remapped.add(matched_current)
            self.topology_manager._overfitting_modules = remapped

        if return_report:
            return report
        return {}

    def enable_diagnostics(self, enable: bool = True) -> None:
        self._enable_diagnostics = enable

    def _record_diagnostic(self, step: int, adjust_results: Dict[str, Any],
                           topology_info: Optional[Dict[str, Any]]) -> None:
        if not self._enable_diagnostics:
            return

        entry = {
            "step": step,
            "num_groups": self.topology_manager.num_groups(),
            "module_stats": {},
            "merged_pairs": [],
            "split_groups": [],
            "merged_details": [],
            "split_details": [],
        }

        group_map = self.topology_manager.get_module_group_map()
        for name in self.module_names:
            adj = adjust_results.get(name, {})
            group = self.topology_manager.get_group(name)
            if group is not None:
                alpha, beta = group.alpha.item(), group.beta.item()
                group_mean = alpha / (alpha + beta)
                group_var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
            else:
                group_mean = adj.get("current_mean", 0.0)
                group_var = adj.get("current_var", 0.0)

            entry["module_stats"][name] = {
                "smoothing_mean": group_mean,
                "smoothing_var": group_var,
                "group_id": group_map.get(name),
                "confidence": self.smoothing_modules[name].ema_confidence.item(),
                "marginal_entropy": self.smoothing_modules[name].ema_marginal_entropy.item(),
            }

        if topology_info is not None:
            entry["merged_pairs"] = [
                {"group_a": int(p[0]), "group_b": int(p[1])}
                for p in topology_info.get("merged_pairs", [])
            ]
            entry["split_groups"] = [int(g) for g in topology_info.get("split_groups", [])]
            entry["merged_details"] = list(topology_info.get("merged_details", []))
            entry["split_details"] = list(topology_info.get("split_details", []))

        self._diagnostic_log.append(entry)

    def get_diagnostic_log(self) -> List[Dict[str, Any]]:
        return list(self._diagnostic_log)

    def export_diagnostics_json(self) -> str:
        def _convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item()
            if isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        log_data = _convert(self._diagnostic_log)
        return json.dumps(log_data, indent=2, ensure_ascii=False)

    def save_diagnostics_json(self, filepath: str) -> None:
        json_str = self.export_diagnostics_json()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)

    def export_diagnostics_csv(self) -> str:
        output = io.StringIO()
        fieldnames = [
            "step", "num_groups", "module_name", "group_id",
            "smoothing_mean", "smoothing_var",
            "confidence", "marginal_entropy",
            "merged_from_group", "split_from_group"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for entry in self._diagnostic_log:
            step = entry["step"]
            num_groups = entry["num_groups"]

            merged_map = {}
            for mp in entry.get("merged_pairs", []):
                merged_map[mp["group_b"]] = mp["group_a"]

            split_groups = set(entry.get("split_groups", []))

            for mod_name, stats in entry["module_stats"].items():
                row = {
                    "step": step,
                    "num_groups": num_groups,
                    "module_name": mod_name,
                    "group_id": stats.get("group_id", -1),
                    "smoothing_mean": f"{stats.get('smoothing_mean', 0):.6f}",
                    "smoothing_var": f"{stats.get('smoothing_var', 0):.8f}",
                    "confidence": f"{stats.get('confidence', 0):.6f}",
                    "marginal_entropy": f"{stats.get('marginal_entropy', 0):.6f}",
                    "merged_from_group": "",
                    "split_from_group": "",
                }

                gid = stats.get("group_id")
                if gid in merged_map:
                    row["merged_from_group"] = str(merged_map[gid])
                if gid in split_groups:
                    row["split_from_group"] = str(gid)

                writer.writerow(row)

        return output.getvalue()

    def save_diagnostics_csv(self, filepath: str) -> None:
        csv_str = self.export_diagnostics_csv()
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_str)

    def get_merge_split_events(self,
                               min_step: Optional[int] = None,
                               max_step: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        events = {
            "merge_events": [],
            "split_events": [],
        }

        for entry in self._diagnostic_log:
            step = entry["step"]
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue

            merged_details = entry.get("merged_details", [])
            if merged_details:
                for md in merged_details:
                    events["merge_events"].append({
                        "step": step,
                        "group_a": md["group_a"],
                        "group_b": md["group_b"],
                        "modules": sorted(md["modules"]),
                    })
            else:
                module_group_map_prev = None
                if "module_stats" in entry:
                    module_group_map_prev = {}
                    for m, s in entry["module_stats"].items():
                        module_group_map_prev[m] = s.get("group_id", -1)

                for mp in entry.get("merged_pairs", []):
                    involved_modules = []
                    if module_group_map_prev is not None:
                        gid_a = mp["group_a"]
                        gid_b = mp["group_b"]
                        for m, gid in module_group_map_prev.items():
                            if gid in (gid_a, gid_b):
                                involved_modules.append(m)

                    events["merge_events"].append({
                        "step": step,
                        "group_a": mp["group_a"],
                        "group_b": mp["group_b"],
                        "modules": sorted(involved_modules),
                    })

            split_details = entry.get("split_details", [])
            if split_details:
                for sd in split_details:
                    events["split_events"].append({
                        "step": step,
                        "group_id": sd["group_id"],
                        "modules": sorted(sd["modules"]),
                    })
            else:
                module_group_map_prev = None
                if "module_stats" in entry:
                    module_group_map_prev = {}
                    for m, s in entry["module_stats"].items():
                        module_group_map_prev[m] = s.get("group_id", -1)

                for sg in entry.get("split_groups", []):
                    involved_modules = []
                    if module_group_map_prev is not None:
                        for m, gid in module_group_map_prev.items():
                            if gid == sg:
                                involved_modules.append(m)

                    events["split_events"].append({
                        "step": step,
                        "group_id": sg,
                        "modules": sorted(involved_modules),
                    })

        return events

    def _filter_log_by_steps(self,
                             min_step: Optional[int],
                             max_step: Optional[int]) -> List[Dict[str, Any]]:
        if min_step is None and max_step is None:
            return self._diagnostic_log

        result = []
        for entry in self._diagnostic_log:
            s = entry["step"]
            if min_step is not None and s < min_step:
                continue
            if max_step is not None and s > max_step:
                continue
            result.append(entry)
        return result

    def get_module_summary(self,
                           min_step: Optional[int] = None,
                           max_step: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        summaries = {}

        events = self.get_merge_split_events(min_step, max_step)

        mod_merge_count: Dict[str, int] = {n: 0 for n in self.module_names}
        mod_split_count: Dict[str, int] = {n: 0 for n in self.module_names}
        mod_merge_steps: Dict[str, List[int]] = {n: [] for n in self.module_names}
        mod_split_steps: Dict[str, List[int]] = {n: [] for n in self.module_names}

        for ev in events["merge_events"]:
            for m in ev["modules"]:
                if m in mod_merge_count:
                    mod_merge_count[m] += 1
                    mod_merge_steps[m].append(ev["step"])

        for ev in events["split_events"]:
            for m in ev["modules"]:
                if m in mod_split_count:
                    mod_split_count[m] += 1
                    mod_split_steps[m].append(ev["step"])

        filtered_log = self._filter_log_by_steps(min_step, max_step)

        for name in self.module_names:
            module = self.smoothing_modules[name]
            group_id = self.topology_manager.get_module_group_map().get(name)

            means = []
            vars_ = []
            confidences = []
            entropies = []
            group_ids = []

            for entry in filtered_log:
                if "module_stats" in entry and name in entry["module_stats"]:
                    stats = entry["module_stats"][name]
                    means.append(stats.get("smoothing_mean", 0.0))
                    vars_.append(stats.get("smoothing_var", 0.0))
                    confidences.append(stats.get("confidence", 0.0))
                    entropies.append(stats.get("marginal_entropy", 0.0))
                    group_ids.append(stats.get("group_id", -1))

            group_transitions = []
            prev_gid = None
            for gid in group_ids:
                if prev_gid is not None and gid != prev_gid:
                    group_transitions.append({
                        "from_group": prev_gid,
                        "to_group": gid,
                    })
                prev_gid = gid

            n = len(means)
            num_stages = 3 if n >= 6 else 1 if n > 0 else 0
            stage_stats = []
            if n >= 3:
                if n >= 9:
                    stage_sizes = [n // 3] * 3
                    for i in range(n % 3):
                        stage_sizes[i] += 1
                else:
                    stage_sizes = [max(1, n // 3)] * 3
                    total = sum(stage_sizes)
                    stage_sizes[-1] += (n - total)

                idx = 0
                for si, sz in enumerate(stage_sizes):
                    seg = means[idx:idx + sz]
                    vars_seg = vars_[idx:idx + sz]
                    if seg:
                        stage_stats.append({
                            "stage": si,
                            "size": sz,
                            "avg_smoothing_mean": sum(seg) / len(seg),
                            "max_smoothing_var": max(vars_seg) if vars_seg else 0.0,
                        })
                    idx += sz

            summary = {
                "module_name": name,
                "current_group_id": group_id,
                "current_smoothing_mean": module.smoothing_mean.item(),
                "current_smoothing_var": module.smoothing_var.item(),
                "current_confidence": module.ema_confidence.item(),
                "current_marginal_entropy": module.ema_marginal_entropy.item(),
                "num_samples": n,
                "avg_smoothing_mean": sum(means) / len(means) if means else 0.0,
                "max_smoothing_mean": max(means) if means else 0.0,
                "min_smoothing_mean": min(means) if means else 0.0,
                "max_smoothing_var": max(vars_) if vars_ else 0.0,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "avg_marginal_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
                "merge_count": mod_merge_count[name],
                "split_count": mod_split_count[name],
                "merge_steps": mod_merge_steps[name],
                "split_steps": mod_split_steps[name],
                "group_transitions": group_transitions,
                "stage_stats": stage_stats,
                "is_frozen": self.is_frozen(name),
            }
            summaries[name] = summary

        return summaries

    def get_smoothing_curves(self,
                             min_step: Optional[int] = None,
                             max_step: Optional[int] = None) -> Dict[str, Any]:
        filtered_log = self._filter_log_by_steps(min_step, max_step)

        steps = []
        num_groups = []

        per_module = {}
        for name in self.module_names:
            per_module[name] = {
                "smoothing_mean": [],
                "smoothing_var": [],
                "confidence": [],
                "marginal_entropy": [],
                "group_id": [],
            }

        for entry in filtered_log:
            steps.append(entry["step"])
            num_groups.append(entry.get("num_groups", 0))

            for name, stats in entry.get("module_stats", {}).items():
                if name in per_module:
                    per_module[name]["smoothing_mean"].append(stats.get("smoothing_mean", 0.0))
                    per_module[name]["smoothing_var"].append(stats.get("smoothing_var", 0.0))
                    per_module[name]["confidence"].append(stats.get("confidence", 0.0))
                    per_module[name]["marginal_entropy"].append(stats.get("marginal_entropy", 0.0))
                    per_module[name]["group_id"].append(stats.get("group_id", -1))

        events = self.get_merge_split_events(min_step, max_step)

        curves = {
            "steps": steps,
            "num_groups": num_groups,
            "per_module": per_module,
            "merge_events": events["merge_events"],
            "split_events": events["split_events"],
        }

        return curves

    def extra_repr(self) -> str:
        return (f"num_classes={self.num_classes}, "
                f"num_modules={len(self.module_names)}, "
                f"num_groups={self.topology_manager.num_groups()}")

    # ------------------------------------------------------------------
    # 设备迁移辅助
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        if len(self.smoothing_modules) > 0:
            first = next(iter(self.smoothing_modules.values()))
            return first.log_alpha.device
        return torch.device("cpu")

    def to_device(self, device):
        return self.to(torch.device(device))

    # ------------------------------------------------------------------
    # 分布式训练接入
    # ------------------------------------------------------------------
    @staticmethod
    def is_distributed() -> bool:
        try:
            import torch.distributed as dist
            return dist.is_available() and dist.is_initialized()
        except Exception:
            return False

    @staticmethod
    def get_rank() -> int:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    @staticmethod
    def get_world_size() -> int:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_world_size()
        except Exception:
            pass
        return 1

    def _dist_all_reduce_mean_(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed():
            return tensor
        try:
            import torch.distributed as dist
            world = self.get_world_size()
            if world > 1:
                tensor = tensor.clone()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor = tensor / world
            return tensor
        except Exception:
            return tensor

    def _dist_all_reduce_dict_means(self, value_dict: Dict[str, Any]
                                     ) -> Dict[str, Any]:
        if not self.is_distributed() or self.get_world_size() <= 1:
            return value_dict
        reduced: Dict[str, Any] = {}
        for k, v in value_dict.items():
            if isinstance(v, torch.Tensor):
                reduced[k] = self._dist_all_reduce_mean_(v)
            elif isinstance(v, (int, float)):
                t = torch.tensor(float(v), device=self.device)
                reduced[k] = float(self._dist_all_reduce_mean_(t).item())
            else:
                reduced[k] = v
        return reduced

    def _dist_broadcast_topology_decision(self, topology_info: Optional[Dict[str, Any]]
                                           ) -> Optional[Dict[str, Any]]:
        if not self.is_distributed() or self.get_world_size() <= 1:
            return topology_info
        try:
            import torch.distributed as dist
            rank = self.get_rank()

            if topology_info is None:
                serialized: bytes = b""
            else:
                import pickle
                sanitized = {}
                for k, v in topology_info.items():
                    if isinstance(v, dict):
                        sanitized[k] = {kk: (vv.item() if isinstance(vv, torch.Tensor) else vv)
                                       for kk, vv in v.items()}
                    elif isinstance(v, (list, tuple)):
                        sanitized[k] = [(x.item() if isinstance(x, torch.Tensor) else x) for x in v]
                    elif isinstance(v, torch.Tensor):
                        sanitized[k] = v.item()
                    else:
                        sanitized[k] = v
                serialized = pickle.dumps(sanitized)

            length_t = torch.tensor(len(serialized), dtype=torch.long, device=self.device)
            dist.broadcast(length_t, src=0)
            target_len = int(length_t.item())

            if rank == 0:
                data_t = torch.frombuffer(bytearray(serialized), dtype=torch.uint8).to(self.device)
            else:
                data_t = torch.zeros(target_len, dtype=torch.uint8, device=self.device)

            if data_t.numel() < target_len:
                pad = torch.zeros(target_len - data_t.numel(), dtype=torch.uint8, device=self.device)
                data_t = torch.cat([data_t, pad])

            dist.broadcast(data_t, src=0)

            import pickle
            bytes_data = bytes(data_t.cpu().numpy().tobytes()[:target_len])
            if len(bytes_data) == 0:
                return None
            recovered = pickle.loads(bytes_data)
            return recovered
        except Exception:
            return topology_info

    def aggregate_module_stats_distributed(self) -> Dict[str, Dict[str, float]]:
        """将各 rank 上的置信度、边缘熵做 all-reduce mean，返回全局值 dict。"""
        local: Dict[str, Dict[str, float]] = {}
        for name in self.module_names:
            mod = self.smoothing_modules[name]
            local[name] = {
                "confidence": float(mod.ema_confidence.item()),
                "entropy": float(mod.ema_marginal_entropy.item()),
            }
        if not self.is_distributed() or self.get_world_size() <= 1:
            return local

        reduced = {}
        for name, d in local.items():
            conf_t = torch.tensor(d["confidence"], device=self.device)
            ent_t = torch.tensor(d["entropy"], device=self.device)
            conf_g = float(self._dist_all_reduce_mean_(conf_t).item())
            ent_g = float(self._dist_all_reduce_mean_(ent_t).item())
            reduced[name] = {"confidence": conf_g, "entropy": ent_g}
        return reduced

    def build_distributed_flat_log(self,
                                   local_flat_log: Dict[str, Any]
                                   ) -> Dict[str, Any]:
        """在 local_flat_log 基础上加上 global_ 前缀的全局指标。"""
        result: Dict[str, Any] = {"local/" + k: v for k, v in local_flat_log.items()}
        result.update(local_flat_log)

        if not self.is_distributed() or self.get_world_size() <= 1:
            for k, v in local_flat_log.items():
                if isinstance(v, (int, float)):
                    result["global/" + k] = float(v)
                elif isinstance(v, torch.Tensor):
                    result["global/" + k] = float(v.item())
                else:
                    result["global/" + k] = v
            result["global/dist_rank"] = 0
            result["global/dist_world_size"] = 1
            return result

        result["global/dist_rank"] = self.get_rank()
        result["global/dist_world_size"] = self.get_world_size()

        numeric_keys = [k for k, v in local_flat_log.items()
                        if isinstance(v, (int, float, torch.Tensor))]
        local_vals = []
        for k in numeric_keys:
            v = local_flat_log[k]
            if isinstance(v, torch.Tensor):
                local_vals.append(float(v.item()))
            else:
                local_vals.append(float(v))
        stacked = torch.tensor(local_vals, device=self.device)
        global_vals = self._dist_all_reduce_mean_(stacked)

        for k, gv in zip(numeric_keys, global_vals.tolist()):
            result["global/" + k] = float(gv)

        for k in local_flat_log:
            if k not in numeric_keys:
                result["global/" + k] = local_flat_log[k]

        return result
