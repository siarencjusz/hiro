"""Example of an image preprocessing for preparation of a training dataset"""
import jax
import jax.numpy as jnp
import pymongo
from tqdm import tqdm
from pymongo import MongoClient
import dm_pix as pix

from constants import TRAIN_VALID_SPLIT, DATASET_NAME, RANDOM_CROP_COUNT, TRAIN_PIXELS
from utils import get_secret, load_image, jarray2json


def color_range(img):
    """
    Calculate the maximum distance between any 2 pixels as a L2 for all three RGB channels
    :param img: image in a shape of (width, height, channel)
    :return: L2 distance between any 2 pixels of the input image
    """
    flat_img = img.reshape(-1, 3)
    return jnp.max(jnp.sum(
        (flat_img[jnp.newaxis, :, :] - flat_img[:, jnp.newaxis, :]) ** 2,
        axis=2), axis=(0, 1))


def preprocess_imgs(input_imgs: pymongo.collection, img_names: list,
                    color_range_threshold: float = 0.05) -> jnp.ndarray:
    """
    Preprocess input images in mongodb into a jnp array for training
    :param input_imgs: mongodb collection containing images jpg streams
    :param img_names: _id of mongodb collection to be preprocessed
    :param color_range_threshold: a threshold of a minimum color range for ready image to be
    accepted for a dataset
    :return: jnp array of preprocessed images ready for training
    """
    preprocessed_imgs = []
    for img_name in tqdm(img_names, unit='img', desc='Preprocessing img'):
        org_img = load_image(input_imgs.find_one({'_id': img_name})['content'])
        img = jax.image.resize(org_img, jnp.array(org_img.shape) // jnp.array([2, 2, 1]),
                               method='lanczos5')
        img -= jnp.minimum(0, jnp.min(img))
        img /= jnp.maximum(1, jnp.max(img))
        crop_keys = jax.random.randint(key=jax.random.PRNGKey(123), shape=[RANDOM_CROP_COUNT],
                                       minval=0, maxval=1e6)
        for crop_key in crop_keys:
            crop_img = pix.random_crop(key=jax.random.PRNGKey(crop_key), image=img,
                                       crop_sizes=(TRAIN_PIXELS, TRAIN_PIXELS, 3))
            if color_range(crop_img) > color_range_threshold ** 2:
                preprocessed_imgs.append(crop_img)

    return jnp.stack(preprocessed_imgs, axis=0)


def upload_dataset_to_mongo(mongo_collection, dataset_name, data):
    """
    Upload dataset into a mongodb collection storing each sample as a separate document
    :param mongo_collection: Mongodb collection for uploading
    :param dataset_name: name of the dataset to be uploaded, sample id will be attached
    :param data: array for upload to a mongodb
    """
    for sample_id, sample in tqdm(enumerate(data), unit='sample',
                                  desc=f'Upload {dataset_name} to mongodb'):
        serialized_sample = jarray2json(sample, f'{dataset_name}_{sample_id}')
        mongo_collection.insert_one(serialized_sample)


if __name__ == '__main__':

    # Input data to be preprocessed are stored in input_imgs collection
    mongo = MongoClient(get_secret('mongo_connection_string'))
    img_ids = mongo.hiro.input_imgs.distinct('_id')

    # Input images are split into train a valid subsets
    train_imgs_names = img_ids[:int(TRAIN_VALID_SPLIT * len(img_ids))]
    train_imgs = preprocess_imgs(mongo.hiro.input_imgs, train_imgs_names)
    upload_dataset_to_mongo(mongo.hiro[DATASET_NAME], 'train', train_imgs)

    valid_imgs_names = img_ids[int(TRAIN_VALID_SPLIT * len(img_ids)):]
    valid_imgs = preprocess_imgs(mongo.hiro.input_imgs, valid_imgs_names)
    upload_dataset_to_mongo(mongo.hiro[DATASET_NAME], 'valid', valid_imgs)
