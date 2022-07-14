"""Example of an image preprocessing for preparation of a training dataset"""
import jax.numpy as jnp
import pymongo
from tqdm import tqdm
from jax import random
from jax import image as jimage
from pymongo import MongoClient
import dm_pix as pix
from utils import get_secret, load_image, serialize_jarray_for_mongo


def preprocess_imgs(input_imgs: pymongo.collection, img_names: list) -> jnp.ndarray:
    """
    Preprocess input images in mongodb into a jnp array for training
    :param input_imgs: mongodb collection containing images jpg streams
    :param img_names: _id of mongodb collection to be preprocessed
    :return: jnp array of preprocessed images ready for training
    """
    preprocessed_imgs = []
    for img_name in tqdm(img_names, unit='img', desc='Preprocessing img'):
        org_img = load_image(input_imgs.find_one({'_id': img_name})['content'])
        img = jimage.resize(org_img, jnp.array(org_img.shape) // jnp.array([2, 2, 1]),
                            method='lanczos5')
        img -= jnp.minimum(0, jnp.min(img))
        img /= jnp.maximum(1, jnp.max(img))
        crop_keys = random.randint(key=random.PRNGKey(123), shape=[RANDOM_CROP_COUNT],
                                   minval=0, maxval=1e6)
        for crop_key in crop_keys:
            crop_img = pix.random_crop(key=random.PRNGKey(crop_key), image=img,
                                       crop_sizes=(TRAIN_PIXELS, TRAIN_PIXELS, 3))
            preprocessed_imgs.append(crop_img)

    return jnp.stack(preprocessed_imgs, axis=0)


if __name__ == '__main__':

    TRAIN_PIXELS = 128  # width and height of training image
    RANDOM_CROP_COUNT = 16
    TRAIN_VALID_SPLIT = 0.8
    DATASET_VERSION = 1  # Used to generate dataset id

    mongo = MongoClient(get_secret('mongo_connection_string'))
    img_ids = mongo.hiro.input_imgs.distinct('_id')
    train_imgs_names = img_ids[:int(TRAIN_VALID_SPLIT * len(img_ids))]
    valid_imgs_names = img_ids[int(TRAIN_VALID_SPLIT * len(img_ids)):]

    train_imgs = preprocess_imgs(mongo.hiro.input_imgs, train_imgs_names)
    valid_imgs = preprocess_imgs(mongo.hiro.input_imgs, valid_imgs_names)

    mongo.hiro.preprocessed_imgs.insert_one(
        serialize_jarray_for_mongo(train_imgs, f'train_{DATASET_VERSION}'))
    mongo.hiro.preprocessed_imgs.insert_one(
        serialize_jarray_for_mongo(train_imgs, f'valid_{DATASET_VERSION}'))
