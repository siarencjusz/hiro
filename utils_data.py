"""Utilities for database and data access"""
from jax import numpy as jnp
from tqdm import tqdm


def get_secret(secret_name, secret_dir='./secrets/'):
    """
    Read secret from specified location as a string
    :param secret_name: name of a secret to be read
    :param secret_dir: folder where secrets are kept
    :return: first line of a secret file content
    """
    with open(f'{secret_dir}{secret_name}', 'r', encoding="utf8") as secret_file:
        return secret_file.readline().strip()


def download_dataset(mongo_collection, sample_regex):
    """
    Download dataset from mongodb collection
    :param mongo_collection: mongodb collection containing a dataset
    :param sample_regex: regex expression for basic filtering of a dataset
    :return: stacked along first axis samples downloaded from a mongodb
    """
    return jnp.stack([json2jarray(sample)
                      for sample in tqdm(mongo_collection.find({'_id': {'$regex': sample_regex}}),
                                         desc=f'Download {sample_regex} samples from mongodb',
                                         unit='sample',)], axis=0)


def jarray2json(jarray: jnp.ndarray, mongo_id: str = None) -> dict:
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


def json2jarray(serialized_jarray: dict) -> jnp.ndarray:
    """
    Deserialize mongodb object into a jax array restoring its content, dtype and shape
    :param serialized_jarray: an object returned from find or findOne mongo collection
    :return: deserialized jax array
    """
    return jnp.frombuffer(serialized_jarray['content'],
                          dtype=serialized_jarray['dtype']).reshape(serialized_jarray['shape'])


def nested_jsons2jarrays(jarrays_tree: dict):
    """
    Deserialize nested mongodb objects into a jax arrays for each tree leaf restoring
    its content, dtype and shape
    :param jarrays_tree: jarrays in a nested tree which stores mongo serialized jarrays
    :return: deserialized jax arrays tree
    """
    # At leaf a serialized array is expected
    if set(jarrays_tree.keys()) == {'_id', 'content', 'dtype', 'shape'}:
        result = json2jarray(jarrays_tree)
    # Otherwise progress with tree traversal until leaf is found
    else:
        result = {k: nested_jsons2jarrays(v) for k, v in jarrays_tree.items()}
    return result
