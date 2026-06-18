import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class BetaSmoothModule(nn.Module):
    def __init__(self, module_name: str, num_classes: int,
                 init_alpha: float = 2.0, init_beta: float = 10.0,
                 learnable_params: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.module_name = module_name
        self.num_classes = num_classes
        self.eps = eps
        self.learnable_params = learnable_params

        if learnable_params:
            self.log_alpha = nn.Parameter(torch.tensor(init_alpha).log())
            self.log_beta = nn.Parameter(torch.tensor(init_beta).log())
        else:
            self.register_buffer("log_alpha", torch.tensor(init_alpha).log())
            self.register_buffer("log_beta", torch.tensor(init_beta).log())

        self.register_buffer("_ema_confidence", torch.tensor(1.0 / num_classes))
        self.register_buffer("_ema_marginal_entropy", torch.tensor(0.0))
        self._ema_momentum = 0.9

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp() + self.eps

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp() + self.eps

    @property
    def smoothing_mean(self) -> torch.Tensor:
        a = self.alpha
        b = self.beta
        return a / (a + b)

    @property
    def smoothing_var(self) -> torch.Tensor:
        a = self.alpha
        b = self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    @property
    def ema_confidence(self) -> torch.Tensor:
        return self._ema_confidence

    @property
    def ema_marginal_entropy(self) -> torch.Tensor:
        return self._ema_marginal_entropy

    def sample_smoothing(self, batch_size: int = 1) -> torch.Tensor:
        a = self.alpha
        b = self.beta
        beta_dist = Beta(a, b)
        return beta_dist.sample((batch_size,))

    def compute_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        confidence = probs.max(dim=-1)[0].mean()
        return confidence

    def compute_marginal_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        marginal = probs.mean(dim=0)
        marginal = torch.clamp(marginal, self.eps, 1.0 - self.eps)
        entropy = -(marginal * marginal.log()).sum()
        return entropy

    def update_stats(self, probs: torch.Tensor) -> None:
        with torch.no_grad():
            confidence = self.compute_confidence(probs.detach())
            marginal_entropy = self.compute_marginal_entropy(probs.detach())

            self._ema_confidence = (
                self._ema_momentum * self._ema_confidence
                + (1 - self._ema_momentum) * confidence
            )
            self._ema_marginal_entropy = (
                self._ema_momentum * self._ema_marginal_entropy
                + (1 - self._ema_momentum) * marginal_entropy
            )

    def smooth_labels(self, labels: torch.Tensor, probs: torch.Tensor,
                      use_mean: bool = False) -> torch.Tensor:
        if use_mean:
            smoothing = self.smoothing_mean
        else:
            if labels.dim() == 1:
                smoothing = self.sample_smoothing(1).squeeze()
            else:
                smoothing = self.sample_smoothing(1).squeeze()

        if labels.dim() == 1:
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            one_hot = labels.float()

        uniform = torch.ones_like(one_hot) / self.num_classes
        smooth_labels = (1.0 - smoothing) * one_hot + smoothing * uniform

        return smooth_labels

    def adjust_beta_params(self, target_mean: float, target_var: float,
                           lr: float = 0.01) -> None:
        with torch.no_grad():
            current_mean = self.smoothing_mean.item()
            current_var = self.smoothing_var.item()

            mean_error = target_mean - current_mean
            var_error = target_var - current_var

            a = self.alpha.item()
            b = self.beta.item()

            if a + b > 0:
                d_mean_da = b / ((a + b) ** 2)
                d_mean_db = -a / ((a + b) ** 2)

                denom = (a + b) ** 3 * (a + b + 1) ** 2
                d_var_da = (b * (b - a) * (a + b + 1) - a * b * (a + b + 1) - a * b * (a + b)) / denom if denom != 0 else 0
                d_var_db = (a * (a - b) * (a + b + 1) - a * b * (a + b + 1) - a * b * (a + b)) / denom if denom != 0 else 0

                det = d_mean_da * d_var_db - d_mean_db * d_var_da

                if abs(det) > 1e-10:
                    da = (d_var_db * mean_error - d_mean_db * var_error) / det
                    db = (-d_var_da * mean_error + d_mean_da * var_error) / det

                    new_a = max(0.1, a + lr * da)
                    new_b = max(0.1, b + lr * db)

                    self.log_alpha.data = torch.tensor(new_a).log().to(self.log_alpha.device)
                    self.log_beta.data = torch.tensor(new_b).log().to(self.log_beta.device)

    def extra_repr(self) -> str:
        return (f"module_name={self.module_name}, num_classes={self.num_classes}, "
                f"learnable_params={self.learnable_params}")
