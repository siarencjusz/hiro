"""Example of an image preprocessing for preparation of a training dataset"""


import jax.numpy as jnp
from pymongo import MongoClient

import dm_pix as pix
from matplotlib import pyplot as plt
from utils import get_secret, load_image

if __name__ == '__main__':

    mongo = MongoClient(get_secret('mongo_connection_string'))
    raw_img = mongo.hiro.input_imgs.find_one()
    img = load_image(raw_img['content'])
    print(img.shape)
    plt.imshow(img)
    plt.show()
