import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import copy

from .beta_smoothing import BetaSmoothModule


@dataclass
class SmoothGroup:
    group_id: int
    module_names: List[str]
    alpha: torch.Tensor
    beta: torch.Tensor
    merge_count: int = 0
    split_count: int = 0

    def __post_init__(self):
        self.alpha = nn.Parameter(self.alpha.clone().detach())
        self.beta = nn.Parameter(self.beta.clone().detach())


class TopologyManager(nn.Module):
    def __init__(self,
                 num_classes: int,
                 merge_threshold: float = 0.05,
                 merge_window: int = 10,
                 min_group_size: int = 2,
                 max_groups: int = 8,
                 eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.merge_threshold = merge_threshold
        self.merge_window = merge_window
        self.min_group_size = min_group_size
        self.max_groups = max_groups
        self.eps = eps

        self._groups: Dict[int, SmoothGroup] = {}
        self._module_to_group: Dict[str, int] = {}
        self._module_history: Dict[str, List[float]] = {}
        self._next_group_id: int = 0
        self._step_count: int = 0

        self._overfitting_modules: Set[str] = set()

    def register_module(self, module_name: str,
                        init_alpha: float = 2.0,
                        init_beta: float = 10.0) -> None:
        if module_name in self._module_to_group:
            return

        group_id = self._next_group_id
        self._next_group_id += 1

        alpha = torch.tensor(init_alpha)
        beta = torch.tensor(init_beta)

        group = SmoothGroup(
            group_id=group_id,
            module_names=[module_name],
            alpha=alpha,
            beta=beta
        )
        self._groups[group_id] = group
        self._module_to_group[module_name] = group_id
        self._module_history[module_name] = []

    def get_module_params(self, module_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        group_id = self._module_to_group.get(module_name)
        if group_id is None:
            raise ValueError(f"Module {module_name} not registered")
        group = self._groups[group_id]
        return group.alpha, group.beta

    def get_group(self, module_name: str) -> Optional[SmoothGroup]:
        group_id = self._module_to_group.get(module_name)
        if group_id is None:
            return None
        return self._groups.get(group_id)

    def get_all_groups(self) -> Dict[int, SmoothGroup]:
        return self._groups.copy()

    def get_module_group_map(self) -> Dict[str, int]:
        return self._module_to_group.copy()

    def record_smoothing_value(self, module_name: str, smoothing_mean: float) -> None:
        if module_name not in self._module_history:
            self._module_history[module_name] = []

        self._module_history[module_name].append(smoothing_mean)

        if len(self._module_history[module_name]) > self.merge_window:
            self._module_history[module_name] = self._module_history[module_name][-self.merge_window:]

        self._step_count += 1

    def _avg_smoothing(self, module_name: str) -> float:
        history = self._module_history.get(module_name, [])
        if not history:
            return 0.0
        return sum(history) / len(history)

    def _are_modules_close(self, name1: str, name2: str) -> bool:
        avg1 = self._avg_smoothing(name1)
        avg2 = self._avg_smoothing(name2)
        return abs(avg1 - avg2) < self.merge_threshold

    def mark_overfitting(self, module_name: str) -> None:
        self._overfitting_modules.add(module_name)

    def clear_overfitting_marks(self) -> None:
        self._overfitting_modules.clear()

    def try_merge_groups(self, force_merge: bool = False) -> List[Tuple[int, int]]:
        merged_pairs = []

        if len(self._groups) <= 1:
            return merged_pairs

        changed = True
        while changed:
            changed = False
            group_ids = list(self._groups.keys())

            for i in range(len(group_ids)):
                if changed:
                    break
                for j in range(i + 1, len(group_ids)):
                    gid1, gid2 = group_ids[i], group_ids[j]

                    if gid1 not in self._groups or gid2 not in self._groups:
                        continue

                    g1 = self._groups[gid1]
                    g2 = self._groups[gid2]

                    if self._groups_are_close(g1, g2):
                        merged_pairs.append((gid1, gid2))
                        self._merge_two_groups(gid1, gid2)
                        changed = True
                        break

        return merged_pairs

    def _groups_are_close(self, g1: 'SmoothGroup', g2: 'SmoothGroup') -> bool:
        for m1 in g1.module_names:
            if not self._has_enough_history(m1):
                return False
            for m2 in g2.module_names:
                if not self._has_enough_history(m2):
                    return False
                if not self._are_modules_close(m1, m2):
                    return False
        return True

    def _has_enough_history(self, module_name: str) -> bool:
        history = self._module_history.get(module_name, [])
        return len(history) >= self.merge_window

    def _merge_two_groups(self, gid1: int, gid2: int) -> None:
        g1 = self._groups[gid1]
        g2 = self._groups[gid2]

        size1 = len(g1.module_names)
        size2 = len(g2.module_names)
        total = size1 + size2

        new_alpha = (g1.alpha * size1 + g2.alpha * size2) / total
        new_beta = (g1.beta * size1 + g2.beta * size2) / total

        merged_modules = g1.module_names + g2.module_names

        new_group = SmoothGroup(
            group_id=gid1,
            module_names=merged_modules,
            alpha=new_alpha.detach(),
            beta=new_beta.detach(),
            merge_count=g1.merge_count + g2.merge_count + 1
        )

        for mod_name in merged_modules:
            self._module_to_group[mod_name] = gid1

        self._groups[gid1] = new_group
        del self._groups[gid2]

    def try_split_groups(self) -> List[int]:
        split_groups = []

        for gid, group in list(self._groups.items()):
            if len(group.module_names) < self.min_group_size:
                continue

            has_overfit = any(m in self._overfitting_modules for m in group.module_names)
            if not has_overfit:
                continue

            self._split_group(gid)
            split_groups.append(gid)

        return split_groups

    def _split_group(self, gid: int) -> None:
        group = self._groups[gid]

        if len(group.module_names) <= 1:
            return

        base_alpha = group.alpha.detach()
        base_beta = group.beta.detach()

        for idx, mod_name in enumerate(group.module_names):
            if idx == 0:
                new_gid = gid
                new_group = SmoothGroup(
                    group_id=new_gid,
                    module_names=[mod_name],
                    alpha=base_alpha.clone(),
                    beta=base_beta.clone(),
                    split_count=group.split_count + 1
                )
                self._groups[new_gid] = new_group
            else:
                new_gid = self._next_group_id
                self._next_group_id += 1

                alpha_noise = base_alpha * (1.0 + 0.1 * torch.randn(()))
                beta_noise = base_beta * (1.0 + 0.1 * torch.randn(()))

                new_group = SmoothGroup(
                    group_id=new_gid,
                    module_names=[mod_name],
                    alpha=alpha_noise.detach(),
                    beta=beta_noise.detach(),
                    split_count=group.split_count + 1
                )
                self._groups[new_gid] = new_group

            self._module_to_group[mod_name] = new_gid

    def update_group_params(self, module_name: str,
                            target_alpha: torch.Tensor,
                            target_beta: torch.Tensor,
                            lr: float = 0.01) -> None:
        group_id = self._module_to_group.get(module_name)
        if group_id is None:
            return

        group = self._groups[group_id]

        with torch.no_grad():
            group.alpha.data = group.alpha.data + lr * (target_alpha - group.alpha.data)
            group.beta.data = group.beta.data + lr * (target_beta - group.beta.data)

    def num_groups(self) -> int:
        return len(self._groups)

    def get_group_sizes(self) -> Dict[int, int]:
        return {gid: len(g.module_names) for gid, g in self._groups.items()}

    def reset_module_history(self, module_name: str) -> None:
        if module_name in self._module_history:
            self._module_history[module_name] = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        groups_data = {}
        for gid, group in self._groups.items():
            groups_data[gid] = {
                'group_id': group.group_id,
                'module_names': list(group.module_names),
                'alpha': group.alpha.data.clone(),
                'beta': group.beta.data.clone(),
                'merge_count': group.merge_count,
                'split_count': group.split_count,
            }

        state[prefix + 'topology_groups'] = groups_data
        state[prefix + 'module_to_group'] = dict(self._module_to_group)
        state[prefix + 'module_history'] = {k: list(v) for k, v in self._module_history.items()}
        state[prefix + 'next_group_id'] = self._next_group_id
        state[prefix + 'step_count'] = self._step_count
        state[prefix + 'overfitting_modules'] = list(self._overfitting_modules)

        return state

    def load_state_dict(self, state_dict, strict=True):
        keys_to_extract = [
            'topology_groups', 'module_to_group', 'module_history',
            'next_group_id', 'step_count', 'overfitting_modules'
        ]
        extracted = {}
        filtered_state = {}
        for k, v in state_dict.items():
            base_key = k.rsplit('.', 1)[-1] if '.' in k else k
            if base_key in keys_to_extract:
                extracted[base_key] = v
            else:
                filtered_state[k] = v

        super().load_state_dict(filtered_state, strict=strict)

        if 'topology_groups' in extracted:
            self._groups = {}
            for gid, gdata in extracted['topology_groups'].items():
                group = SmoothGroup(
                    group_id=gdata['group_id'],
                    module_names=list(gdata['module_names']),
                    alpha=gdata['alpha'].clone(),
                    beta=gdata['beta'].clone(),
                    merge_count=gdata.get('merge_count', 0),
                    split_count=gdata.get('split_count', 0),
                )
                self._groups[gid] = group

        if 'module_to_group' in extracted:
            self._module_to_group = dict(extracted['module_to_group'])

        if 'module_history' in extracted:
            self._module_history = {k: list(v) for k, v in extracted['module_history'].items()}

        if 'next_group_id' in extracted:
            self._next_group_id = extracted['next_group_id']

        if 'step_count' in extracted:
            self._step_count = extracted['step_count']

        if 'overfitting_modules' in extracted:
            self._overfitting_modules = set(extracted['overfitting_modules'])

        return {}

    def extra_repr(self) -> str:
        return (f"num_classes={self.num_classes}, merge_threshold={self.merge_threshold}, "
                f"merge_window={self.merge_window}, num_groups={len(self._groups)}")
