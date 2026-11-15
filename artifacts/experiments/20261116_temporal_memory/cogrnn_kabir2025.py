# Vendored verbatim (below this header) from github.com/cogneuroai/RL-with-scale-invariant-memory, commit
# 878f985572bfa9100e3afcd3168bb26678158b2f, file interval_timing3D/CogRNN.py: the CogRNN of Kabir, Mochizuki-Freeman
# & Tiganj, "Deep reinforcement learning with time-scale invariant memory", AAAI 2025 (arXiv 2412.15292). Used here only
# to build the Laplace decay rates s and the Post-inverse matrix, and to check our filtering against its own scan.
from typing import Optional

import torch
import torch.nn as nn
import math

DEBUG_USE_LOG_INV = False  # use more stable (and slower) log implementation


class CogRNN(nn.Module):
    def __init__(
            self,
            tstr_min: float,
            tstr_max: float,
            n_taus: int,
            k: int,
            dt: float,
            g: int,
            DEBUG_dt_scale: float = 1.0,
            batch_first: bool = False,
            linear_scaling_flag: bool = False,
    ):
        """
        Constructs a series of nodes that provide a log-spaced reconstruction of input.
        Args:
            tstr_min:
                Peak time of the first node in the sequence.
            tstr_max:
                Peak time of the last node in the sequence.
            n_taus:
                Number of nodes in the inverse Laplace transform.
            k:
                Order of the derivative in inverse Laplace transform, larger k
                causes more narrow peaks in f_til.
            dt:
                Time step of the simulation.
            g:
                Amplitude scaling of nodes in f_til.
                g=1 -> equal amplitude
                g=0 -> power-law decay of amplitude
            DEBUG_dt_scale:
                Factor to scale dt. Used to correct errors in taustar peaks when k is large.
            linear_scaling_flag:
                Scales the s values linearly instead of logarithmically if True
        """
        super().__init__()
        if tstr_min is None:
            tstr_min = dt
        self.tstr_min = tstr_min
        self.tstr_max = tstr_max
        self.n_taus = n_taus
        self.k = k
        self.dt = dt
        self.g = g
        self.DEBUG_dt_scale = DEBUG_dt_scale
        self.batch_first = batch_first
        self.linear_scaling_flag = linear_scaling_flag
        self.N = n_taus + (2 * k)

        if not self.linear_scaling_flag:
            self.c = c = (tstr_max / tstr_min) ** (1.0 / (n_taus - 1))
            # log spacing constant
            tau_stars_full = tstr_min * torch.logspace(-k, (n_taus - 1 + k), self.N, base=c)
        else:
            self.c = None
            lin_sp = (tstr_max - tstr_min) / (n_taus + k - 1)  # linear spacing between nodes
            tau_stars_full = torch.linspace(tstr_min, tstr_max + (k * lin_sp), self.N)

        self.tau_stars = tau_stars_full[k:-k]

        s_full = k / tau_stars_full  # decay constants
        self.register_buffer("s_full", s_full, persistent=False)

        deriv_matrix = torch.zeros((self.N, self.N))
        if not self.linear_scaling_flag:
            for i in range(1, self.N - 1):
                deriv_matrix[i, i - 1] = -(1 / c) / (s_full[i + 1] - s_full[i - 1])
                deriv_matrix[i, i] = ((1 / c) - c) / (s_full[i + 1] - s_full[i - 1])
                deriv_matrix[i, i + 1] = c / (s_full[i + 1] - s_full[i - 1])
        else:
            for i in range(1, self.N - 1):
                sp = s_full[i + 1]
                si = s_full[i]
                sm = s_full[i - 1]
                deriv_matrix[i, i - 1] = -(sp - si) / (si - sm) / (sp - sm)
                deriv_matrix[i, i] = (((sp - si) / (si - sm)) - ((si - sm) / (sp - si))) / (sp - sm)
                deriv_matrix[i, i + 1] = (si - sm) / (sp - si) / (sp - sm)

        # -1^k * 1/(k!) * s^(k+1) * tau^g * (D^k)F
        post_1 = ((-1) ** k) * deriv_matrix.matrix_power(k).T * (tau_stars_full ** g)
        log_post_2 = -math.lgamma(k + 1) + (k + 1) * s_full.log()  # - math.log(k)

        self.register_buffer("log_post_2", log_post_2, persistent=False)

        if DEBUG_USE_LOG_INV:
            log_deriv_pos = torch.clamp(post_1, min=0).log()
            log_deriv_neg = torch.clamp(-post_1, min=0).log()
            self.register_buffer(
                "log_post_pos", (log_post_2 + log_deriv_pos), persistent=False
            )
            self.register_buffer(
                "log_post_neg", (log_post_2 + log_deriv_neg), persistent=False
            )
        else:
            post = post_1 * log_post_2.exp()
            self.register_buffer("post", post, persistent=False)

        self.F = None
        self.F_full = None
        self.til_f = None

    def forward(
            self,
            f: torch.Tensor,
            h: Optional[torch.Tensor] = None,
            alpha: Optional[torch.Tensor] = None,
            delta: Optional[torch.Tensor] = None,
            return_sequences: bool = True,
            init: bool = False
    ):
        #     f.shape: (seq_len, batch, feature)
        # alpha.shape: (seq_len, batch, feature)
        # delta.shape: (seq_len, batch, feature)
        #
        # if return_sequences is True:
        #         F.shape: (seq_len, batch, feature, s)
        #     til_f.shape: (seq_len, batch, feature, taustar)
        # else:
        #         F.shape: (1, batch, feature, s)
        #     til_f.shape: (1, batch, feature, taustar)

        alpha = alpha if alpha is not None else torch.ones_like(f)
        delta = delta if delta is not None else torch.zeros_like(f)

        if self.batch_first:
            f = f.permute(1, 0, 2)
            alpha = alpha.permute(1, 0, 2)
            delta = delta.permute(1, 0, 2)

        # Generate h0
        if h is None:
            batch_size, feature_size = f.size()[1:]
            h = f.new_zeros((batch_size, feature_size, self.N))
            h = h.log()

        # Laplace
        log_F_full, h = self._laplace(f, h, alpha, delta, return_sequences)

        # Inverse Laplace
        self.til_f = self._invert(log_F_full)

        # Unflatten
        self.F = log_F_full.exp()[:, :, :, self.k: -self.k]

        if self.batch_first:
            self.F = self.F.permute(1, 0, 2, 3)
            self.til_f = self.til_f.permute(1, 0, 2, 3)

        return self.til_f, h, self.F

    def _get_log_laplace_matrix(self, f_size, alpha, delta):
        # alpha.shape: (seq_len, batch, feature) or (batch, feature) or (feature,) or scalar
        # delta.shape: (seq_len, batch, feature) or (batch, feature) or (feature,) or scalar
        #
        # Returns: Laplace transform matrices (with alpha and delta), per timestep

        trans = (alpha * self.dt) + delta
        trans = trans.expand(f_size)
        return torch.einsum("zbf, s -> zbfs", trans, -self.s_full * self.DEBUG_dt_scale)

    def _laplace(self, f, log_F, alpha, delta, return_sequences):
        log_lap_matrices = self._get_log_laplace_matrix(f.size(), alpha, delta)

        log_F_full = []

        for log_lap, log_f in zip(log_lap_matrices, (f * self.dt).unsqueeze(-1).log()):
            log_F = torch.logaddexp(log_F + log_lap, log_f.nan_to_num())

            if return_sequences:
                log_F_full.append(log_F)

        if return_sequences:
            log_F_full = torch.stack(log_F_full)
        else:
            log_F_full = log_F.unsqueeze(0)

        return log_F_full, log_F

    def _invert(self, log_F_full):
        # Inverse Laplace transform
        #
        # log_F_full.shape: (seq_len, batch, feature, s_full)
        # returns (seq_len, batch, feature, taustar)

        if DEBUG_USE_LOG_INV:
            log_pos = logmatmulexp(log_F_full, self.log_post_pos)
            log_neg = logmatmulexp(log_F_full, self.log_post_neg)
            til_f = logsubexp(log_pos, log_neg).exp()
        else:
            til_f = log_F_full.exp() @ self.post
        return til_f[:, :, :, self.k: -self.k]

    def get_translations(self, log_F):
        # generates a series of memory projections for specified timestep
        # log_F.shape: (batch, feature, s_full)
        # returns (batch, feature, delta, taustar)

        deltas = self.tau_stars
        R = torch.outer(deltas, -self.s_full * self.DEBUG_dt_scale).exp()

        delta_F = torch.einsum("ds, bfs -> bfds", R, log_F.exp())
        delta_til_f = self._invert(delta_F.log())
        return delta_til_f

    def extra_repr(self):
        return (
            f"tstr_min={self.tstr_min}, tstr_max={self.tstr_max}, "
            f"n_taus={self.n_taus}, k={self.k}, dt={self.dt}, "
            f"g={self.g}, DEBUG_dt_scale={self.DEBUG_dt_scale}, "
            f"batch_first={self.batch_first}"
        )

    def check(self):
        """Verify model is working as expected"""
        seq_len = int(self.tstr_max / self.dt) * 3
        f = torch.zeros(seq_len, 1, 1)
        f[0, :, :] = 1

        h = f.new_zeros((*f.size()[1:], self.N)).log()

        log_F_full, h = self._laplace(f, h, torch.ones_like(f), torch.zeros_like(f), True)
        til_f = self._invert(log_F_full)

        peaks = []
        for idx in range(self.n_taus):
            peaks.append(til_f[:, 0, 0, idx].argmax() / (self.tau_stars[idx] / self.dt))

        debug_dt_calc = torch.tensor(peaks).mean().item() * self.DEBUG_dt_scale
        print(f"Corrected DEBUG_dt_scale: {debug_dt_calc}")
        print(f"k: {self.k}\tmax: {til_f.max()}")
        if (til_f < 0).any():
            print("Negative values produced in output!")


def logmatmulexp(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    assert log_a.shape[-1] == log_b.shape[-2]
    b = torch.broadcast_shapes(log_a.shape[:-2], log_b.shape[:-2])
    log_a = log_a.expand(*b, -1, -1)
    log_b = log_b.expand(*b, -1, -1)

    return torch.logsumexp(log_a[..., :, :, None] + log_b[..., None, :, :], dim=-2)

def logsubexp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + (-torch.expm1(y - x)).log().nan_to_num(nan=float("-inf"))