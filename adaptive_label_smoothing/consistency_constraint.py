import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class TopologyConsistencyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 consistency_weight: float = 0.1,
                 distance_type: str = "cosine",
                 eps: float = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.consistency_weight = consistency_weight
        self.distance_type = distance_type
        self.eps = eps

    def _pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        if self.distance_type == "euclidean":
            x_norm = (x ** 2).sum(dim=1, keepdim=True)
            distances = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
            distances = torch.clamp(distances, min=0.0)
            distances = torch.sqrt(distances + self.eps)
            return distances

        elif self.distance_type == "cosine":
            x_normalized = F.normalize(x, p=2, dim=1, eps=self.eps)
            similarity = torch.mm(x_normalized, x_normalized.t())
            similarity = torch.clamp(similarity, -1.0, 1.0)
            distances = 1.0 - similarity
            return distances

        elif self.distance_type == "kl":
            x_clamped = torch.clamp(x, min=self.eps)
            x_log = torch.log(x_clamped)
            x_expanded = x_clamped.unsqueeze(1)
            x_log_expanded = x_log.unsqueeze(0)

            kl_distances = (x_expanded * (torch.log(x_expanded + self.eps) - x_log_expanded)).sum(dim=-1)
            kl_distances_t = (x_clamped.unsqueeze(0) * (torch.log(x_clamped.unsqueeze(0) + self.eps) - x_log.unsqueeze(1))).sum(dim=-1)
            distances = (kl_distances + kl_distances_t) / 2.0
            distances = torch.clamp(distances, min=0.0)
            return distances

        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def _rank_correlation(self, dist1: torch.Tensor, dist2: torch.Tensor) -> torch.Tensor:
        n = dist1.size(0)
        if n <= 1:
            return torch.tensor(1.0, device=dist1.device)

        triu_idx = torch.triu_indices(n, n, offset=1)
        d1_vec = dist1[triu_idx[0], triu_idx[1]]
        d2_vec = dist2[triu_idx[0], triu_idx[1]]

        rank1 = torch.argsort(torch.argsort(d1_vec)).float()
        rank2 = torch.argsort(torch.argsort(d2_vec)).float()

        rank1_mean = rank1.mean()
        rank2_mean = rank2.mean()

        covariance = ((rank1 - rank1_mean) * (rank2 - rank2_mean)).mean()
        std1 = rank1.std() + self.eps
        std2 = rank2.std() + self.eps

        correlation = covariance / (std1 * std2)
        return correlation

    def _distance_correlation(self, dist1: torch.Tensor, dist2: torch.Tensor) -> torch.Tensor:
        n = dist1.size(0)
        if n <= 1:
            return torch.tensor(0.0, device=dist1.device)

        a = dist1.clone()
        b = dist2.clone()

        row_mean_a = a.mean(dim=1, keepdim=True)
        col_mean_a = a.mean(dim=0, keepdim=True)
        mean_a = a.mean()
        a_centered = a - row_mean_a - col_mean_a + mean_a

        row_mean_b = b.mean(dim=1, keepdim=True)
        col_mean_b = b.mean(dim=0, keepdim=True)
        mean_b = b.mean()
        b_centered = b - row_mean_b - col_mean_b + mean_b

        d_cov_sq = (a_centered * b_centered).mean()
        d_var1_sq = (a_centered ** 2).mean() + self.eps
        d_var2_sq = (b_centered ** 2).mean() + self.eps

        d_cor = torch.sqrt(torch.abs(d_cov_sq) / (torch.sqrt(d_var1_sq) * torch.sqrt(d_var2_sq) + self.eps))
        return d_cor

    def pairwise_consistency_loss(self, labels1: torch.Tensor,
                                  labels2: torch.Tensor) -> torch.Tensor:
        dist1 = self._pairwise_distances(labels1)
        dist2 = self._pairwise_distances(labels2)

        dist1_flat = dist1.view(-1)
        dist2_flat = dist2.view(-1)

        dist1_norm = F.normalize(dist1_flat.unsqueeze(0), p=2, dim=1, eps=self.eps).squeeze()
        dist2_norm = F.normalize(dist2_flat.unsqueeze(0), p=2, dim=1, eps=self.eps).squeeze()

        similarity = (dist1_norm * dist2_norm).sum()
        similarity = torch.clamp(similarity, -1.0, 1.0)

        loss = 1.0 - similarity
        loss = torch.clamp(loss, min=0.0)

        return loss

    def forward(self, module_smooth_labels: Dict[str, torch.Tensor],
                reference_module: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        if len(module_smooth_labels) < 2:
            return torch.tensor(0.0), {}

        module_names = list(module_smooth_labels.keys())
        total_loss = 0.0
        pair_count = 0
        consistency_scores = {}

        if reference_module is not None and reference_module in module_smooth_labels:
            ref_labels = module_smooth_labels[reference_module]

            for name in module_names:
                if name == reference_module:
                    continue

                labels = module_smooth_labels[name]
                pair_loss = self.pairwise_consistency_loss(ref_labels, labels)
                total_loss = total_loss + pair_loss
                pair_count += 1

                score = 1.0 - pair_loss.item()
                consistency_scores[f"{reference_module}_vs_{name}"] = score
        else:
            for i in range(len(module_names)):
                for j in range(i + 1, len(module_names)):
                    name1, name2 = module_names[i], module_names[j]
                    labels1 = module_smooth_labels[name1]
                    labels2 = module_smooth_labels[name2]

                    pair_loss = self.pairwise_consistency_loss(labels1, labels2)
                    total_loss = total_loss + pair_loss
                    pair_count += 1

                    score = 1.0 - pair_loss.item()
                    consistency_scores[f"{name1}_vs_{name2}"] = score

        if pair_count > 0:
            avg_loss = total_loss / pair_count
        else:
            avg_loss = torch.tensor(0.0)

        weighted_loss = self.consistency_weight * avg_loss

        return weighted_loss, consistency_scores

    def compute_topology_preservation(self, original_labels: torch.Tensor,
                                      smooth_labels: torch.Tensor) -> float:
        dist_orig = self._pairwise_distances(original_labels)
        dist_smooth = self._pairwise_distances(smooth_labels)

        rank_corr = self._rank_correlation(dist_orig, dist_smooth)
        return rank_corr.item()

    def extra_repr(self) -> str:
        return (f"num_classes={self.num_classes}, "
                f"consistency_weight={self.consistency_weight}, "
                f"distance_type={self.distance_type}")
