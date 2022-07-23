"""Example of an image preprocessing for preparation of a training dataset"""
from io import BytesIO

import jax
import jax.numpy as jnp
import pymongo
from tqdm import tqdm
from pymongo import MongoClient
import dm_pix as pix
from PIL import Image as pil

from constants import TRAIN_VALID_SPLIT, DATASET_NAME, RANDOM_CROP_COUNT, TRAIN_PIXELS, \
    SCALING_FACTORS
from utils import get_secret, jarray2json, prepare_input_up_scaled_img


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

        for main_scale in SCALING_FACTORS:
            # Goal of the preprocessing pipeline is to produce a training dataset
            # therefore a maximum quality should be kept for a reference image.
            # A common preprocessing is established to ensure similar quality for all input images.
            # Down and up scaling should be clearly defined in order for later details reproduction
            raw_org_img = pil.open(BytesIO(input_imgs.find_one({'_id': img_name})['content']))

            raw_org_img_array = jnp.array(raw_org_img) / 255.0
            org_img = jax.image.resize(raw_org_img_array, jnp.array(raw_org_img_array.shape)
                                       // jnp.array([main_scale, main_scale, 1]), method='lanczos5')

            up_scaled_input_img = prepare_input_up_scaled_img(org_img)
            both_img = jnp.concatenate((org_img, up_scaled_input_img), axis=-1)

            # For training both high and low quality are stored together as a separate channels
            crop_key = jax.random.PRNGKey(123)
            for _ in tqdm(range(RANDOM_CROP_COUNT // (main_scale ** 2)),
                          desc=f'Cropping with scale = {main_scale}', unit='img'):
                crop_key, subkey = jax.random.split(crop_key)
                crop_img = pix.random_crop(key=subkey, image=both_img,
                                           crop_sizes=(TRAIN_PIXELS, TRAIN_PIXELS, 6))
                if color_range(crop_img[..., :3]) > color_range_threshold ** 2:
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
