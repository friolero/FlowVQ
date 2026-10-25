import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import manifold


def get_vis_embedding(raw_embedding, codebook_embedding):
    """Joint 3-D t-SNE embedding of latents and codes (no-op for <= 3-D inputs)."""
    if raw_embedding.shape[1] > 3:
        combined = np.vstack([raw_embedding, codebook_embedding])
        reduced = manifold.TSNE(n_components=3).fit_transform(combined)
        return reduced[: raw_embedding.shape[0]], reduced[raw_embedding.shape[0] :]
    else:
        return raw_embedding, codebook_embedding


def vis_3d_codebook(raw_zes, raw_vocab_t, n_vis=1000, logger=None):
    """3-D t-SNE plot of latents (o) and codes (^), separately and overlaid.

    raw_zes: (B, C, H, W) or (B, D) latents; raw_vocab_t: (K, D) codebook.
    Both are subsampled to min(#latents, #codes, n_vis) points.
    """
    zes_vis = raw_zes.permute(0, 2, 3, 1).flatten(0, -2) if raw_zes.ndim == 4 else raw_zes
    if raw_vocab_t is None:
        raw_vocab_t = zes_vis
    n = min(zes_vis.size(0), raw_vocab_t.size(0), n_vis)
    ze_vis = zes_vis[torch.randperm(zes_vis.size(0))][:n].cpu().numpy()
    vocab_vis = raw_vocab_t[torch.randperm(raw_vocab_t.size(0))][:n].cpu().numpy()
    ze_vis, vocab_vis = get_vis_embedding(ze_vis, vocab_vis)

    x_min, y_min, z_min = ze_vis.min(axis=0)
    x_max, y_max, z_max = ze_vis.max(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(ze_vis[:, 0], ze_vis[:, 1], ze_vis[:, 2], marker="o")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(vocab_vis[:, 0], vocab_vis[:, 1], vocab_vis[:, 2], marker="^")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(ze_vis[:, 0], ze_vis[:, 1], ze_vis[:, 2], marker="o")
    ax.scatter(vocab_vis[:, 0], vocab_vis[:, 1], vocab_vis[:, 2], marker="^")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig
