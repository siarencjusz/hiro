"""Run trained model through an example image"""
from jax import numpy as jnp
import haiku as hk
from pymongo import MongoClient
from tqdm import tqdm
from PIL import Image as pil
from matplotlib import pyplot as plt

from cnn_model import CNN
from constants import EXPERIMENT_NAME, TRAIN_PIXELS, CNN_LAYERS
from utils_data import get_secret, nested_jsons2jarrays
from utils_img import calculate_img_residual


if __name__ == '__main__':

    INPUT_IMAGE_FILE_NAME = './secrets/test.jpg'
    EXPERIMENT_TIMESTAMP = "2022-07-19 22:26:03.798270"
    mongo = MongoClient(get_secret('mongo_connection_string'))
    doc = mongo.hiro.models.find_one({'training_timestamp': EXPERIMENT_TIMESTAMP,
                                      'experiment_name': EXPERIMENT_NAME,
                                      'epoch': 48})
    params = nested_jsons2jarrays(doc['params'])

    def conv_net_fun(model_in):
        """
        Functional wrapper for cnn model
        :param model_in: model input
        """
        cnn = CNN(doc['architecture'])
        return cnn(model_in)
    conv_net = hk.transform(conv_net_fun)

    img_org = jnp.array(pil.open(INPUT_IMAGE_FILE_NAME)) / 255.
    img_res = calculate_img_residual(img_org[jnp.newaxis, ...])[0]
    scaled_residual = jnp.minimum(jnp.maximum(-0.5, img_res / 0.2), 0.5) + 0.5
    img_down = img_org - img_res
    in_padded = jnp.pad(img_down, ((CNN_LAYERS, CNN_LAYERS),
                                   (CNN_LAYERS, CNN_LAYERS), (0, 0)), mode='edge')

    lines = []
    for x in tqdm(range(in_padded.shape[1] + 1 - TRAIN_PIXELS)):
        line = jnp.stack([conv_net.apply(
            params, None, in_padded[n:(n+TRAIN_PIXELS), x:(x+TRAIN_PIXELS)])
            for n in range(in_padded.shape[0] + 1 - TRAIN_PIXELS)], axis=0)
        lines.append(line[:, CNN_LAYERS, CNN_LAYERS])

    model_res = jnp.stack(lines, axis=1)
    scaled_model_res = jnp.minimum(jnp.maximum(-0.5, model_res / 0.2), 0.5) + 0.5

    plt.imshow(scaled_model_res)
    plt.show()

    plt.imshow(img_down + model_res)
    plt.show()
