import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import json
import csv
import io

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

        self._diagnostic_log: List[Dict[str, Any]] = []
        self._enable_diagnostics: bool = True

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

    def compute_losses(self,
                       module_logits: Dict[str, torch.Tensor],
                       labels: torch.Tensor,
                       module_val_accs: Optional[Dict[str, float]] = None,
                       use_group_params: bool = True,
                       reduction: str = "mean") -> Dict[str, Any]:
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
        per_module_loss = {}
        total_cls_loss = 0.0

        for name in self.module_names:
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

            per_module_loss[name] = loss
            total_cls_loss = total_cls_loss + loss

        if reduction == "mean" and len(per_module_loss) > 0:
            avg_cls_loss = total_cls_loss / len(per_module_loss)
        else:
            avg_cls_loss = total_cls_loss

        consistency_loss = train_result["consistency_loss"]
        total_loss = avg_cls_loss + consistency_loss

        result = {
            "total_loss": total_loss,
            "classification_loss": avg_cls_loss,
            "consistency_loss": consistency_loss,
            "per_module_loss": per_module_loss,
            "smooth_labels": smooth_labels_dict,
            "consistency_scores": train_result["consistency_scores"],
            "adjust_results": train_result["adjust_results"],
            "topology_info": train_result["topology_info"],
            "step": train_result["step"],
            "smoothing_info": self.get_smoothing_info(),
        }

        return result

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        state[prefix + 'module_names'] = list(self.module_names)
        state[prefix + 'num_classes'] = self.num_classes
        state[prefix + 'step_count'] = self._step_count
        state[prefix + 'merge_split_interval'] = self._merge_split_interval
        state[prefix + 'enable_diagnostics'] = self._enable_diagnostics
        state[prefix + 'diagnostic_log'] = list(self._diagnostic_log)

        state[prefix + 'scheduler_val_acc_history'] = dict(self.scheduler._val_acc_history)
        state[prefix + 'scheduler_best_val_acc'] = dict(self.scheduler._best_val_acc)
        state[prefix + 'scheduler_overfit_counter'] = dict(self.scheduler._overfit_counter)

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

    def load_state_dict(self, state_dict, strict=True):
        own_prefix_keys = [
            'module_names', 'num_classes', 'step_count',
            'merge_split_interval', 'enable_diagnostics', 'diagnostic_log',
            'scheduler_val_acc_history', 'scheduler_best_val_acc', 'scheduler_overfit_counter',
            'topology_groups', 'topology_module_to_group', 'topology_module_history',
            'topology_next_group_id', 'topology_step_count', 'topology_overfitting_modules',
        ]

        extracted = {}
        filtered_state = {}

        for k, v in state_dict.items():
            is_own = False
            for ok in own_prefix_keys:
                if k.endswith('.' + ok) or k == ok:
                    extracted[ok] = v
                    is_own = True
                    break
            if not is_own:
                filtered_state[k] = v

        super().load_state_dict(filtered_state, strict=strict)

        if 'module_names' in extracted:
            self.module_names = list(extracted['module_names'])
        if 'step_count' in extracted:
            self._step_count = extracted['step_count']
        if 'merge_split_interval' in extracted:
            self._merge_split_interval = extracted['merge_split_interval']
        if 'enable_diagnostics' in extracted:
            self._enable_diagnostics = extracted['enable_diagnostics']
        if 'diagnostic_log' in extracted:
            self._diagnostic_log = list(extracted['diagnostic_log'])

        if 'scheduler_val_acc_history' in extracted:
            self.scheduler._val_acc_history = dict(extracted['scheduler_val_acc_history'])
        if 'scheduler_best_val_acc' in extracted:
            self.scheduler._best_val_acc = dict(extracted['scheduler_best_val_acc'])
        if 'scheduler_overfit_counter' in extracted:
            self.scheduler._overfit_counter = dict(extracted['scheduler_overfit_counter'])

        from .topology_manager import SmoothGroup
        if 'topology_groups' in extracted:
            groups_data = extracted['topology_groups']
            new_groups = {}
            for gid, gdata in groups_data.items():
                group = SmoothGroup(
                    group_id=gdata['group_id'],
                    module_names=list(gdata['module_names']),
                    alpha=gdata['alpha'].clone(),
                    beta=gdata['beta'].clone(),
                    merge_count=gdata.get('merge_count', 0),
                    split_count=gdata.get('split_count', 0),
                )
                new_groups[gid] = group
            self.topology_manager._groups = new_groups

        if 'topology_module_to_group' in extracted:
            self.topology_manager._module_to_group = dict(extracted['topology_module_to_group'])
        if 'topology_module_history' in extracted:
            self.topology_manager._module_history = {k: list(v) for k, v in extracted['topology_module_history'].items()}
        if 'topology_next_group_id' in extracted:
            self.topology_manager._next_group_id = extracted['topology_next_group_id']
        if 'topology_step_count' in extracted:
            self.topology_manager._step_count = extracted['topology_step_count']
        if 'topology_overfitting_modules' in extracted:
            self.topology_manager._overfitting_modules = set(extracted['topology_overfitting_modules'])

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

    def get_merge_split_events(self) -> Dict[str, List[Dict[str, Any]]]:
        events = {
            "merge_events": [],
            "split_events": [],
        }

        for entry in self._diagnostic_log:
            step = entry["step"]

            for mp in entry.get("merged_pairs", []):
                events["merge_events"].append({
                    "step": step,
                    "group_a": mp["group_a"],
                    "group_b": mp["group_b"],
                })

            for sg in entry.get("split_groups", []):
                events["split_events"].append({
                    "step": step,
                    "group_id": sg,
                })

        return events

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

        self._record_diagnostic(self._step_count, adjust_results, topology_info)

        result = {
            "smooth_labels": smooth_labels_dict,
            "consistency_loss": consistency_loss_val,
            "consistency_scores": consistency_scores,
            "adjust_results": adjust_results,
            "topology_info": topology_info,
            "step": self._step_count
        }

        return result

    def extra_repr(self) -> str:
        return (f"num_classes={self.num_classes}, "
                f"num_modules={len(self.module_names)}, "
                f"num_groups={self.topology_manager.num_groups()}")
