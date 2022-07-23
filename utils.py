"""Set of utilities functions"""
from io import BytesIO
import jax
from jax import numpy as jnp
import numpy as np
from PIL import Image as pil
import dm_pix as pix
from tqdm import tqdm
import matplotlib.pyplot as plt


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
