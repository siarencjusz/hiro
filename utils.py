"""Set of utilities functions"""
from io import BytesIO
from PIL import Image as pil
import jax.numpy as jnp


def get_secret(secret_name, secret_dir='./secrets/'):
    """
    Read secret from specified location as a string
    :param secret_name: name of a secret to be read
    :param secret_dir: folder where secrets are kept
    :return: first line of a secret file content
    """
    with open(f'{secret_dir}{secret_name}', 'r', encoding="utf8") as secret_file:
        return secret_file.readline().strip()


def load_image(raw_buffer: bytes):
    """
    Read image into array from a buffer containing jpeg image
    :param raw_buffer: raw jpg file bytes
    :return: jax array containing the loaded image
    """
    return jnp.array(pil.open(BytesIO(raw_buffer)), dtype=jnp.float32) / 255.
