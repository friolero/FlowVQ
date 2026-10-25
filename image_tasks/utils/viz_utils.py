"""This file contains some utils functions for visualization."""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from einops import rearrange
from PIL import Image
from sklearn import manifold


def make_viz_from_samples(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
) -> Tuple[List[Image.Image], torch.Tensor]:
    """
    Creates a visualization of the samples.

    Args:
        original_images -> torch.Tensor: The original images.
        reconstructed_images -> torch.Tensor: The reconstructed images.

    Returns:
        A tuple of the images for saving and for logging.
    """

    # Convert to PIL images
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    original_images = torch.clamp(original_images, 0.0, 1.0)
    reconstructed_images *= 255.0
    original_images *= 255.0
    reconstructed_images = reconstructed_images.cpu()
    original_images = original_images.cpu()
    difference_images = torch.abs(original_images - reconstructed_images)

    to_stack = [original_images, reconstructed_images, difference_images]

    images_for_logging = rearrange(torch.stack(to_stack), "(l1 l2) b c h w -> b c (l1 h) (l2 w)", l1=1).byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging


def make_viz_reconstructed_stage_two(
    reconstructed: torch.Tensor, predicted: torch.Tensor
) -> Tuple[List[Image.Image], torch.Tensor]:
    """
    Creates a visualization of the reconstructed and predicted samples.

    Args:
        predicted -> torch.Tensor: The predicted images.
        reconstructed -> torch.Tensor: The reconstructed images.
    Returns:
        A tuple of the images for saving and for logging.
    """
    reconstructed = torch.clamp(reconstructed, 0.0, 1.0)
    predicted = torch.clamp(predicted, 0.0, 1.0)

    images_for_logging = torch.cat([reconstructed, predicted], dim=-1) * 255.0
    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging


def make_viz_generated_stage_two(
    generated: torch.Tensor,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Creates a visualization of the generated samples.

    Args:
        generated -> torch.Tensor: The generated images, shape (batch_size, 3, height, width),
            where batch_size is an even number.
    Returns:
        A tuple of the images for saving and for logging.
    """
    generated = torch.clamp(generated, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(generated, "(l1 l2) c h w -> c (l1 h) (l2 w)", l1=2)

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    return images_for_saving, images_for_logging


def get_vis_embedding(raw_embedding, codebook_embedding, method="tsne"):
    assert method in ["tsne", "isomap"], "Incompatible dimension reducing method."
    if raw_embedding.shape[1] > 3:
        combined = np.vstack([raw_embedding, codebook_embedding])
        if method == "isomap":
            dim_reductor = manifold.Isomap(n_neighbors=12, n_components=3, p=1)
        else:
            dim_reductor = manifold.TSNE(n_components=3)
        reduced = dim_reductor.fit_transform(combined)
        return reduced[: raw_embedding.shape[0]], reduced[raw_embedding.shape[0] :]
    else:
        return raw_embedding, codebook_embedding


def vis_3d_codebook(raw_zes, raw_vocab_t, n_vis=1000, logger=None):

    zes_vis = raw_zes.permute(0, 2, 3, 1).flatten(0, -2)
    if raw_vocab_t is None:
        raw_vocab_t = raw_zes
    ze_vis = zes_vis[torch.randperm(zes_vis.size(0))][:n_vis].cpu().numpy()
    #vocab_vis = raw_vocab_t.permute(0, 2, 3, 1).flatten(0, -2)
    vocab_vis = raw_vocab_t[torch.randperm(raw_vocab_t.size(0))][:n_vis].cpu().numpy()
    ze_vis, vocab_vis = get_vis_embedding(ze_vis, vocab_vis)

    x_min, y_min, z_min = ze_vis.min(axis=0)
    x_max, y_max, z_max = ze_vis.max(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(ze_vis[:, 0], ze_vis[:, 1], ze_vis[:, 2], marker="o")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(vocab_vis[:, 0], vocab_vis[:, 1], vocab_vis[:, 2], marker="^")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(ze_vis[:, 0], ze_vis[:, 1], ze_vis[:, 2], marker="o")
    ax.scatter(vocab_vis[:, 0], vocab_vis[:, 1], vocab_vis[:, 2], marker="^")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    return fig
