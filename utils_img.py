"""Set of utilities functions"""
from io import BytesIO
import jax
from jax import numpy as jnp
import numpy as np
from PIL import Image as pil
import dm_pix as pix


def load_image(raw_buffer: bytes):
    """
    Read image into array from a buffer containing jpeg image
    :param raw_buffer: raw jpg file bytes
    :return: jax array containing the loaded image
    """
    return jnp.array(pil.open(BytesIO(raw_buffer)), dtype=jnp.float32) / 255.


def calculate_img_residual(imgs: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate images residuals for recovery by down-sampling then up-sampling input images
    :param imgs: input images for residuals calculation with shape (sample, width, height, channel)
    :return: Residual per each image for details recovery
    """
    imgs_down = jax.image.resize(imgs, jnp.array(imgs.shape) //
                                 jnp.array([1, 2, 2, 1]), method='bicubic')
    imgs_up = jax.image.resize(imgs_down, imgs.shape, method='bicubic')
    return imgs - imgs_up


def prepare_input_up_scaled_img(org_img: jnp.array) -> jnp.array:
    """
    Down scale original images, blur then up scale
    :param org_img: original high fidelity image
    :return: blurred image as expected to be the model input
    """
    # It is assumed an input might be a blurred image, therefore the original one is
    # degraded by jpeg low compression then blurr
    with BytesIO() as low_quality:
        img_array = np.array(jnp.maximum(jnp.minimum(org_img, 1), 0) * 255, dtype='uint8')
        pil.fromarray(img_array).save(low_quality, format='jpeg', quality=20)
        low_quality_buffer = low_quality.getvalue()
    low_quality_img = load_image(low_quality_buffer)
    # Down scale and blurr image, this is the expected model input
    down_scaled_img = jax.image.resize(low_quality_img, jnp.array(low_quality_img.shape)
                                       // jnp.array([2, 2, 1]), method='lanczos5')
    raw_input_img = pix.gaussian_blur(down_scaled_img, sigma=0.6, kernel_size=5)
    # Up scaling is the first input preprocessing step
    up_scaled_input_img = jax.image.resize(raw_input_img, jnp.array(low_quality_img.shape),
                                           method='trilinear')
    return up_scaled_input_img
