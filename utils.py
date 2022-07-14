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


def serialize_jarray_for_mongo(jarray: jnp.ndarray, mongo_id: str) -> dict:
    """
    Serialize input jarray into a json ready for insert into a mongodb collection
    :param jarray: array to be serialized
    :param mongo_id: unique name to be assigned to _id field
    :return: json like object ready for insertion into a mongodb collection
    """
    return {'_id': mongo_id,
            'shape': jarray.shape,
            'content': jarray.flatten().tobytes(),
            'dtype': str(jarray.dtype)}


def deserialize_jarray_from_mongo(serialized_jarray: dict) -> jnp.ndarray:
    """
    Deserialize mongodb object into a jax array restoring its content, dtype and shape
    :param serialized_jarray: an object returned from find or findOne mongo collection
    :return: deserialized jax array
    """
    return jnp.frombuffer(serialized_jarray['content'],
                          dtype=serialized_jarray['dtype']).reshape(serialized_jarray['shape'])
