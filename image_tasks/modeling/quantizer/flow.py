import math
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from modeling.quantizer.quantizer_utils import entropy_loss_fn
from torch import nn
from torch.distributions.normal import Normal
from zuko.utils import odeint


def blocked_min_dist(x, centroids, block_size=1024):
    N = x.shape[0]
    K = centroids.shape[0]

    best_dists = torch.full((N,), float("inf"), device=x.device)
    best_inds = torch.zeros(N, dtype=torch.long, device=x.device)

    for i in range(0, K, block_size):
        c = centroids[i : i + block_size]  # [B, D]

        d = ((x.unsqueeze(1) - c.unsqueeze(0)) ** 2).sum(-1)  # [N, B]

        min_d, min_i = d.min(dim=1)

        mask = min_d < best_dists
        best_dists[mask] = min_d[mask]
        best_inds[mask] = i + min_i[mask]

    return best_inds


def find_nearest(raw_query, dictionary, max_bs=8 * 16 * 16 * 1024 * 10):

    # 8, 10, 16, 16
    #
    if raw_query.ndim == 4:
        flattened_query = (
            rearrange(raw_query, "b c h w -> b h w c")
            .contiguous()
            .view(-1, raw_query.shape[1])
        )
    else:
        flattened_query = raw_query
    dict_size = dictionary.shape[0]

    n_batch = max(1, math.ceil(raw_query.numel() * dict_size / max_bs))
    n_batch = int(n_batch)
    bs = math.ceil(flattened_query.shape[0] / n_batch)

    zq_inds = None
    zqs = None
    for i in range(n_batch):
        query = flattened_query[i * bs : (i + 1) * bs]
        num_query = query.shape[0]
        Q = query.unsqueeze(1).repeat(1, dict_size, 1)
        D = dictionary.unsqueeze(0).repeat(num_query, 1, 1)
        dists = ((Q - D) ** 2).sum(-1)
        _, min_inds = dists.min(1)
        z_q = dictionary[min_inds]
        zq_inds = (
            min_inds if zq_inds is None else torch.cat([zq_inds, min_inds], 0)
        )
        zqs = z_q if zqs is None else torch.cat([zqs, z_q], 0)
    if raw_query.ndim == 4:
        return zq_inds.reshape(
            raw_query.shape[0], raw_query.shape[2], raw_query.shape[3]
        ), zqs.reshape(
            raw_query.shape[0],
            raw_query.shape[2],
            raw_query.shape[3],
            raw_query.shape[1],
        ).permute(
            0, 3, 1, 2
        )
    else:
        return zq_inds, zqs


class IndependentGaussianDist:
    def __init__(self, sample_dim, loc=None, std=None):

        self.sample_dim = sample_dim

        if loc is None:
            self._loc = torch.zeros(self.sample_dim)
        else:
            assert loc.shape[-1] == self.sample_dim, "Wrong loc dimension."
            self._loc = loc

        if std is None:
            self._std = torch.ones(sample_dim)
        else:
            assert std.shape[-1] == self.sample_dim, "Wrong var dimension."
            self._std = std

        self._dist = [
            Normal(self._loc[i], self._std[i]) for i in range(self.sample_dim)
        ]

    def sample(self, n_data):
        x = torch.cat(
            [
                self._dist[i].sample((n_data,)).reshape(-1, 1)
                for i in range(self.sample_dim)
            ],
            -1,
        )
        return x

    def log_prob(self, x, cond):
        logp = torch.cat(
            [
                self._dist[i].log_prob(x[:, i]).reshape(-1, 1)
                for i in range(self.sample_dim)
            ],
            -1,
        ).sum(-1)
        return logp

    def log_prob_at_cdf(self, cdfs):
        logp = torch.cat(
            [
                self._dist[i].log_prob(self._dist[i].icdf(cdfs)).reshape(-1, 1)
                for i in range(self.sample_dim)
            ],
            -1,
        )
        return logp


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self, x_dim, c_dim=None, h_dim=256, n_hidden=3):

        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.t_dim = h_dim
        if self.c_dim is not None:
            input_dim = self.x_dim + self.c_dim + self.t_dim
        else:
            input_dim = self.x_dim + self.t_dim

        self.x_mlp = nn.Linear(x_dim, h_dim)
        self.t_enc = SinusoidalPosEmb(h_dim)
        if c_dim is not None:
            self.c_mlp = nn.Linear(self.c_dim, h_dim)

        self.mid_layer = nn.ModuleList()
        for in_dim, out_dim in zip([h_dim] * n_hidden, [h_dim] * n_hidden):
            self.mid_layer.extend([nn.Linear(in_dim, out_dim), nn.ELU()])
        self.final_layer = nn.Linear(h_dim, self.x_dim)

    def forward(self, x, t, c=None):
        t_emb = self.t_enc(t)
        h = self.x_mlp(x) + t_emb
        if c is not None:
            h += self.c_mlp(c)
        for layer in self.mid_layer:
            h = layer(h)
        output = self.final_layer(h)
        return output


def skewed_timestep_sample(
    num_samples: int, device: torch.device
) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


class FlowMatchingLoss(nn.Module):
    def __init__(self, sigma_min=1e-4, skewed_timestep=False):
        super().__init__()

        self.sigma_min = sigma_min
        self.skewed_timestep = skewed_timestep

    def forward(self, flow, x, z=None, c=None, return_raw=False):
        if self.skewed_timestep:
            t = (
                skewed_timestep_sample(x.flatten(0, -2).shape[0], x.device)
                .reshape(x.shape[:-1])
                .unsqueeze(-1)
            )
        else:
            t = torch.rand_like(x[..., 0, None])
        if z is None:
            z = torch.randn_like(x)
        psi_t = (1 - (1 - self.sigma_min) * t) * z + t * x
        d_psi = x - (1 - self.sigma_min) * z
        pred_psi = flow.net(psi_t, t.squeeze(-1), c=c)
        if return_raw:
            return pred_psi, d_psi
        else:
            return (pred_psi - d_psi.detach()).square().mean()


class DiscreteFlowMatchingLoss(nn.Module):
    def __init__(self, prior_choices, sigma_min=1e-4, skewed_timestep=False):
        super().__init__()

        self.prior_choices = prior_choices
        self.sigma_min = sigma_min
        self.skewed_timestep = skewed_timestep

    def forward(self, flow, x, c=None):
        z_inds = torch.randint(
            low=0, high=self.prior_choices.shape[0], size=(x.shape[0],)
        ).long()
        z = self.prior_choices[z_inds].to(x.device)
        if self.skewed_timestep:
            t = (
                skewed_timestep_sample(x.flatten(0, -2).shape[0], x.device)
                .reshape(x.shape[:-1])
                .unsqueeze(-1)
            )
        else:
            t = torch.rand(x.shape[:-1]).to(x.device).unsqueeze(-1)
        psi_t = (1 - (1 - self.sigma_min) * t) * z + t * x
        d_psi = x - (1 - self.sigma_min) * z
        fml = (flow.net(psi_t, t.squeeze(), c=c) - d_psi.detach()).square()
        return fml.mean()


class ZukoPFlow(nn.Module):
    # def __init__(self, prior_dist, device, t_limit=None, x_limit=None, n_stack=1, c_dim=None, **kwargs):
    def __init__(
        self, net, prior_dist, t_limit=None, x_limit=None, c_dim=None, **kwargs
    ):

        super().__init__()

        self.prior_dist = prior_dist
        self.t_limit = t_limit
        self.x_limit = x_limit
        self.net = net

    def forward(self, t, x, cond=None):
        # if self.t_limit is not None:
        #    t = torch.clip(t, self.t_limit[0], self.t_limit[1])
        # if self.x_limit is not None:
        #    x = torch.clip(x, self.x_limit[0], self.x_limit[1])
        # t = t.unsqueeze(0).repeat(x.shape[0])
        # return self.net(x, t, cond), torch.zeros_like(cond)
        if len(t.shape) == 0:
            t = t.unsqueeze(0).repeat(x.shape[0]).to(x)
        if self.t_limit is not None:
            t = torch.clip(t, self.t_limit[0], self.t_limit[1])
        if cond is not None:
            return self.net(x, t, cond), torch.zeros_like(cond)
        else:
            return self.net(x, t)

    def encode(self, tgt_x):
        return odeint(self, tgt_x, 1.0, 0.0, phi=self.net.parameters())

    def decode(self, src_x):
        return odeint(self, src_x, 0.0, 1.0, phi=self.net.parameters())

    def rollout(
        self, state, prior_logp=None, direction="forward", n_timesteps=None
    ):
        if isinstance(state, tuple) or isinstance(state, list):
            start_x, cond = state
        else:
            start_x = state
            cond = None

        I = torch.eye(
            start_x.shape[-1], dtype=start_x.dtype, device=start_x.device
        )
        I = I.expand(*start_x.shape, start_x.shape[-1]).movedim(-1, 0)

        def ode_step(t, x, ladj):
            with torch.enable_grad():
                x = x.requires_grad_()
                if cond is None:
                    dt_x = self.forward(t, x)
                else:
                    dt_x = self.forward(t, x, cond)
                jacobian = torch.autograd.grad(
                    dt_x, x, I, create_graph=True, is_grads_batched=True
                )
                trace = torch.einsum("i...i", jacobian)
                if torch.isnan(dt_x).any() or torch.isnan(trace).any():
                    ipdb.set_trace()
                return dt_x, -trace

        if direction == "forward":
            if prior_logp is None:
                log_p0 = d_logp = self.prior_dist.log_prob(start_x, cond).to(
                    start_x
                )
            else:
                log_p0 = d_logp = prior_logp.reshape(-1).to(start_x)
            end_x, log_prob = odeint(
                ode_step, (start_x, d_logp), 0.0, 1.0, phi=self.net.parameters()
            )
            return (start_x, end_x), (log_p0, log_prob)
        else:
            # TODO: integral from 1 is automatically backward fwd_step?
            d_logp = torch.zeros_like(start_x[..., 0])
            end_x, log_prob = odeint(
                ode_step, (start_x, d_logp), 1.0, 0.0, phi=self.net.parameters()
            )
            if prior_logp is None:
                log_p0 = self.prior_dist.log_prob(end_x, cond).to(end_x)
            else:
                log_p0 = prior_logp.reshape(-1).to(end_x)
            return (end_x, start_x), (log_p0, log_p0 - log_prob)


class FlowQuantizer(nn.Module):

    def __init__(
        self,
        x_dim,
        k_dim,
        h_dim=256,
        n_hiddens=3,
        eps=5e-3,
        sample_ratio=50,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.1,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
        self_correction: bool = True,
        online_correction: bool = True,
        prior_type: str = "sobol",
    ):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma
        self.self_correction = self_correction
        self.online_correction = online_correction
        assert prior_type in ("sobol", "iid"), prior_type
        self.prior_type = prior_type

        self.k_dim = k_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_hiddens = n_hiddens

        self.prior_dist = IndependentGaussianDist(
            self.x_dim, loc=torch.zeros(self.x_dim), std=torch.ones(self.x_dim)
        )
        net = MLP(x_dim=self.x_dim)
        self.flow = ZukoPFlow(net, self.prior_dist)

        self.embedding = nn.Embedding(self.k_dim, self.x_dim)
        self.sobol = torch.quasirandom.SobolEngine(
            dimension=self.prior_dist.sample_dim, scramble=True
        )

        self.eps = eps
        self.sample_ratio = sample_ratio
        if self.prior_type == "iid":
            # ablation: iid uniform CDF positions instead of the
            # low-discrepancy Sobol sequence (same icdf pipeline below)
            self.sample_cdfs = torch.rand(
                self.sample_ratio * self.k_dim, self.prior_dist.sample_dim
            ).clamp(self.eps, 1 - self.eps)
        else:
            self.sample_cdfs = self.sobol.draw(
                self.sample_ratio * self.k_dim
            ).clamp(self.eps, 1 - self.eps)

        # if self.discrete:
        #    # Draw from the hyper-cube
        #    prior_codebook = 2 * self.sobol.draw(self.k_dim) - 1
        #    self.criterion = DiscreteFlowMatchingLoss(prior_codebook, self.flow)
        # else:
        if self.prior_type == "iid":
            prior_cdfs = torch.rand(self.k_dim, self.prior_dist.sample_dim).clamp(
                1e-6, 1 - 1e-6
            )
        else:
            prior_cdfs = self.sobol.draw(self.k_dim)
        prior_codebook = torch.cat(
            [
                self.prior_dist._dist[i].icdf(prior_cdfs[:, i]).reshape(-1, 1)
                for i in range(self.x_dim)
            ],
            dim=1,
        )
        # else:
        #    gs = round(self.k_dim ** (1 / self.x_dim))
        #    assert k_dim == gs**self.x_dim, "Please input a square number for vocab_size"
        #    cdf_lls = torch.linspace(0, 1, gs + 1)[:-1]
        #    cdf_uls = torch.linspace(0, 1, gs + 1)[1:]
        #    prob_lls = self.prior_dist.log_prob_at_cdf(cdf_lls).exp()
        #    prob_uls = self.prior_dist.log_prob_at_cdf(cdf_uls).exp()
        #    centroid_coords = [(prob_uls[:, i] - prob_lls[:, i]) / (cdf_uls - cdf_lls) for i in range(self.x_dim)]
        #    prior_codebook = torch.cartesian_prod(*centroid_coords)
        self.register_buffer("prior_codebook", prior_codebook)
        self.initialized = False

        # self.init_net = net
        # self.criterion = FlowMatchingLoss(self.flow)

    @property
    def codebook(self):
        return self.embedding.weight.data

    @property
    def codebook_size(self):
        return self.k_dim

    @property
    def vocab_t(self):
        with torch.no_grad():
            posterior_codebook = self.flow.decode(self.prior_codebook)
        return posterior_codebook

    def set_codebook(self, codebook_vectors):
        with torch.no_grad():
            self.embedding.weight.copy_(codebook_vectors)

    def optim_centroids(self, codebook, optim_steps=0, batch_size=10240):

        with torch.no_grad():
            if codebook is None:
                centroids = self.codebook.clone()
            else:
                centroids = codebook.clone()
            n_batch = math.ceil(self.k_dim * self.sample_ratio / batch_size)
            for n_iter in range(optim_steps):
                if n_iter == 0:
                    sample_priors = torch.cat(
                        [
                            self.prior_dist._dist[i]
                            .icdf(self.sample_cdfs[:, i])
                            .reshape(-1, 1)
                            for i in range(self.x_dim)
                        ],
                        dim=1,
                    ).to(centroids.device)
                    samples = [
                        self.flow.decode(
                            sample_priors[i * batch_size : (i + 1) * batch_size]
                        ).detach()
                        for i in range(n_batch)
                    ]
                    samples = torch.cat(samples, 0)

                for i in range(n_batch):
                    tmp_zq_inds, tmp_zq = find_nearest(
                        samples[i * batch_size : (i + 1) * batch_size],
                        centroids,
                    )
                    if i == 0:
                        zq_inds = tmp_zq_inds
                        zq = tmp_zq
                    else:
                        zq_inds = torch.cat([zq_inds, tmp_zq_inds], 0)
                        zq = torch.cat([zq, tmp_zq], 0)
                for i in range(self.k_dim):
                    if (zq_inds == i).sum() > 0:
                        centroids[i] = samples[zq_inds == i].mean(0)
                    else:
                        print(n_iter, i)
            return centroids

    def get_vq_loss(self, ze_batches):
        vq_losses = []
        for z_e in ze_batches:
            z_q = find_nearest(z_e, self.codebook)[1]
            vq_losses.append((z_q - z_e).abs())
        return torch.cat(vq_losses, 0).mean()

    def update_codebook(self, optim_steps=0):
        print("==> Update the codebook.")
        centroids = self.optim_centroids(self.vocab_t, optim_steps=optim_steps)
        self.set_codebook(centroids)
        self.initialized = True

    @torch.no_grad()
    def update_codebook_parallel(
        self, rank, world_size, optim_steps=1, batch_size=10240
    ):
        centroids = self.vocab_t.clone()  # [K, D]
        if self.self_correction:
            K, D = centroids.shape
            device = centroids.device

            total = self.sample_cdfs.shape[0]
            per_rank = math.ceil(total / world_size)
            start = rank * per_rank
            end = min((rank + 1) * per_rank, total)
            local_priors = torch.cat(
                [
                    self.prior_dist._dist[i]
                    .icdf(self.sample_cdfs[start:end, i])
                    .reshape(-1, 1)
                    for i in range(self.x_dim)
                ],
                dim=1,
            ).to(device)
            local_samples = self.flow.decode(
                local_priors
            ).detach()  # [N_local, D]
            for _ in range(optim_steps):
                if local_samples.shape[0] == 0:
                    zq_inds = torch.empty(0, dtype=torch.long, device=device)
                else:
                    zq_inds = blocked_min_dist(
                        local_samples, centroids, block_size=256
                    )  # [N_local]

                code_sum = torch.zeros(K, D, device=device)
                code_count = torch.zeros(K, device=device)
                if local_samples.shape[0] > 0:
                    code_sum.index_add_(0, zq_inds, local_samples)
                    code_count.index_add_(
                        0, zq_inds, torch.ones_like(zq_inds, dtype=torch.float)
                    )
                if world_size > 1:
                    torch.distributed.all_reduce(
                        code_sum, op=torch.distributed.ReduceOp.SUM
                    )
                    torch.distributed.all_reduce(
                        code_count, op=torch.distributed.ReduceOp.SUM
                    )

                new_centroids = centroids.clone()
                mask = code_count > 0
                new_centroids[mask] = code_sum[mask] / code_count[
                    mask
                ].unsqueeze(1)
                centroids = new_centroids

                # centroids[mask] = code_sum[mask] / code_count[mask].unsqueeze(1)

        self.set_codebook(centroids)
        self.initialized = True

    def quantize(self, z, vocab=None):

        if vocab is None:
            vocab = self.embedding.weight

        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.x_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(vocab**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(vocab, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = vocab[min_encoding_indices.flatten()].view(z.shape)

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        z = rearrange(z, "b h w c -> b c h w").contiguous()
        min_encoding_indices = min_encoding_indices.reshape(
            z_q.shape[0], z_q.shape[2], z_q.shape[3]
        )

        return min_encoding_indices, z_q

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Returns the codebook entry for the given indices.

        Args:
            indices -> torch.Tensor: The indices of the codebook entries.

        Returns:
            z_quantized -> torch.Tensor: The codebook entries.
        """
        # get quantized latent vectors
        z_quantized = self.embedding(indices.int())
        return z_quantized

    def forward(self, z, vocab_size=None):

        min_encoding_indices, z_q = self.quantize(z)

        vq_loss = torch.mean((z_q - z.detach()) ** 2)
        fm_loss = self.loss(z.detach())
        if not self.initialized:
            codebook_loss = 0.0 * vq_loss
            commitment_loss = 0.0 * torch.mean((z_q.detach() - z) ** 2)

            min_encoding_indices = (
                torch.ones(z.shape[0], z.shape[2], z.shape[3])
                .long()
                .to(z.device)
            )
            z_q = z.clone()
        else:
            # fm_loss = self.loss(z.detach())
            codebook_loss = vq_loss
            commitment_loss = torch.mean((z_q.detach() - z) ** 2)

        # compute loss for embedding
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            z_flattened = rearrange(
                rearrange(z, "b c h w -> b h w c").contiguous(),
                "b h w c -> (b h w) c",
            )
            embedding = self.embedding.weight
            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(embedding**2, dim=1)
                - 2 * torch.einsum("bd,dn->bn", z_flattened, embedding.T)
            )
            per_sample_entropy, avg_entropy = entropy_loss_fn(
                -1 * d, self.entropy_loss_temperature, self.entropy_gamma
            )
            entropy_loss = self.entropy_loss_weight * (
                per_sample_entropy - avg_entropy
            )

        loss = self.commitment_cost * commitment_loss
        if self.online_correction:
            loss += codebook_loss
        else:
            loss += 0.0 * codebook_loss
        if self.initialized:
            loss += fm_loss
        else:
            loss += 0.0 * fm_loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        result_dict = dict(
            quantizer_loss=loss,
            vq_loss=vq_loss,
            codebook_loss=codebook_loss,
            commitment_loss=self.commitment_cost * commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            fm_loss=fm_loss if self.initialized else 0.0 * fm_loss,
            min_encoding_indices=min_encoding_indices,
        )

        return z_q, result_dict

    def loss(self, ze):
        if ze.ndim == 4:
            flattened_ze = (
                rearrange(ze, "b c h w -> b h w c")
                .contiguous()
                .view(-1, self.x_dim)
            )
        else:
            assert (
                ze.ndim == 2 and ze.shape[1] == self.x_dim
            ), "Wrong input dimension, should be (N, d)"
            flattened_ze = ze
        criterion = FlowMatchingLoss()
        loss = criterion(self.flow, flattened_ze)
        return loss
