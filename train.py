"""Script for training for local testing of the pipeline"""
import jax
from jax import numpy as jnp
import optax
import haiku as hk
from pymongo import MongoClient
from utils import get_secret, deserialize_jarray_from_mongo


class CNN(hk.Module):
    """Simple sequential convolutional neural network"""
    def __init__(self):
        super().__init__(name="CNN")
        self.conv_model = hk.Sequential([
            hk.Conv2D(output_channels=32, kernel_shape=(3, 3), padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(output_channels=32, kernel_shape=(3, 3), padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(output_channels=32, kernel_shape=(3, 3), padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(output_channels=3, kernel_shape=(3, 3), padding="SAME"),
            jax.nn.relu])

    def __call__(self, x_batch):
        return self.conv_model(x_batch)


def calculate_residual(imgs: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate images residuals for recovery by down-sampling then up-sampling input images
    :param imgs: input images for residuals calculation with shape (sample, width, height, channel)
    :return: Residual per each image for details recovery
    """
    imgs_down = jax.image.resize(imgs, jnp.array(train_imgs.shape) //
                                 jnp.array([1, 2, 2, 1]), method='bicubic')
    imgs_up = jax.image.resize(imgs_down, imgs.shape, method='bicubic')
    return imgs - imgs_up


if __name__ == '__main__':

    DATASET_VERSION = 1
    mongo = MongoClient(get_secret('mongo_connection_string'))

    train_imgs = deserialize_jarray_from_mongo(
        mongo.hiro.preprocessed_imgs.find_one({'_id': f'train_{DATASET_VERSION}'}))
    train_out = calculate_residual(train_imgs)
    train_in = train_imgs - train_out

    valid_imgs = deserialize_jarray_from_mongo(
        mongo.hiro.preprocessed_imgs.find_one({'_id': f'valid_{DATASET_VERSION}'}))
    valid_out = calculate_residual(valid_imgs)
    valid_in = valid_imgs - valid_out

    rng = jax.random.PRNGKey(220714)
    EPOCHS = 25
    BATCH_SIZE = 4
    ADAM_LEARNING_RATE = 1e-3


    # pylint: disable=invalid-name
    def conv_net_fun(x):
        """
        Functional wrapper for cnn model
        :type x: input
        """
        cnn = CNN()
        return cnn(x)

    conv_net = hk.transform(conv_net_fun)
    params = conv_net.init(rng, train_in[:2])
    optimizer = optax.adam(learning_rate=ADAM_LEARNING_RATE)
    optimizer_state = optimizer.init(params)

    def mse(model_params, input_data, actual):
        """
        Functional wrapper for haiku model
        :param model_params: model parameters
        :param input_data: input data for a forward pass
        :param actual: ground truth to calculate loss
        :return: mse loss
        """
        predicted = conv_net.apply(model_params, rng, input_data)
        return jnp.mean(actual - predicted) ** 2

    batches = jnp.arange((train_in.shape[0] // BATCH_SIZE) + 1)
    for epoch in range(EPOCHS):

        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch * BATCH_SIZE), int(batch * BATCH_SIZE + BATCH_SIZE)
            else:
                start, end = int(batch * BATCH_SIZE), None

            loss, param_grads = jax.value_and_grad(mse)(params, train_in[start:end],
                                                        train_out[start:end])

            updates, optimizer_state = optimizer.update(param_grads, optimizer_state)
            params = optax.apply_updates(params, updates)

        train_loss = mse(params, train_in, train_out)
        valid_loss = mse(params, valid_in, valid_out)
        print(f'Epoch {epoch}, '
              f'train mse = {jnp.mean(train_loss)}, '
              f'valid mse = {jnp.mean(valid_loss)}')
