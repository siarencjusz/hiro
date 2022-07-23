"""Collection of utilities for plotting images and image based dataset"""
from jax import numpy as jnp
from matplotlib import pyplot as plt


def plot_sample(original, residual, prediction=None, out_file_name=None, max_value=0.2):
    """
    Plot original image, target residual and optionally prediction,
    scaling desired output to a predefined value
    :param original: original image
    :param residual: target residuals
    :param prediction: predicted residuals
    :param max_value: residual from zero up to scale is plot, bigger values are clipped
    :param out_file_name: if provided output is stored instead of plotted
    """
    plt.figure(figsize=(18 if isinstance(prediction, jnp.ndarray) else 12, 6))
    plt.subplot(1, 3 if isinstance(prediction, jnp.ndarray) else 2, 1)
    plt.axis('off')
    plt.imshow(original)
    plt.subplot(1, 3 if isinstance(prediction, jnp.ndarray) else 2, 2)
    scaled_residual = jnp.minimum(jnp.maximum(-0.5, residual / max_value), 0.5) + 0.5
    plt.axis('off')
    plt.imshow(scaled_residual)
    if isinstance(prediction, jnp.ndarray):
        plt.subplot(1, 3, 3)
        scaled_prediction = jnp.minimum(jnp.maximum(-0.5, prediction / max_value), 0.5) + 0.5
        plt.axis('off')
        plt.imshow(scaled_prediction)
    plt.tight_layout()
    if out_file_name:
        plt.savefig(out_file_name)
    else:
        plt.show()


def plot_samples(originals, residuals, predictions, out_file_name=None, max_value=0.2):
    """
    Plot original image, target residual and optionally prediction,
    scaling desired output to a predefined value
    :param originals: original image
    :param residuals: target residuals
    :param predictions: predicted residuals
    :param max_value: residual from zero up to scale is plot, bigger values are clipped
    :param out_file_name: if provided output is stored instead of plotted
    """
    plt.figure(figsize=(18, 6 * originals.shape[0]))

    for sample, (original, residual, target) in enumerate(zip(originals, residuals, predictions)):
        plt.subplot(originals.shape[0], 3, sample * 3 + 1)
        plt.axis('off')
        plt.imshow(original)
        plt.subplot(originals.shape[0], 3, sample * 3 + 2)
        scaled_residual = jnp.minimum(jnp.maximum(-0.5, residual / max_value), 0.5) + 0.5
        plt.axis('off')
        plt.imshow(scaled_residual)

        plt.subplot(originals.shape[0], 3, sample * 3 + 3)
        scaled_prediction = jnp.minimum(jnp.maximum(-0.5, target / max_value), 0.5) + 0.5
        plt.axis('off')
        plt.imshow(scaled_prediction)
    plt.tight_layout()
    if out_file_name:
        plt.savefig(out_file_name)
    else:
        plt.show()


def plot_imgs(imgs, rows, cols):
    """
    Plot input imgs in a defined grid
    :param imgs: images to be plotted in a shape of (sample, width, height, channel)
    :param rows: how many rows of images to be plotted
    :param cols: how many cols of images to be plotted
    """
    plt.figure(figsize=(rows * 2, cols * 2))
    for sample, img in enumerate(imgs):
        plt.subplot(rows, cols, sample + 1)
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    plt.show()
