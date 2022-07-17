"""Script for training for local testing of the pipeline"""
import jax
from jax import numpy as jnp
import optax
import haiku as hk
from pymongo import MongoClient
import wandb
from utils import get_secret, download_dataset, calculate_img_residual


class CNN(hk.Module):
    """Simple sequential convolutional neural network"""
    def __init__(self, arch):
        super().__init__(name="CNN")
        self.conv_model = hk.Sequential(sum([[
            hk.Conv2D(output_channels=arch['cnn_filters'], kernel_shape=(3, 3), padding="SAME"),
            jax.nn.relu] for _ in range(arch['cnn_layers'] - 1)], []) +
            [hk.Conv2D(output_channels=3, kernel_shape=(3, 3), padding="SAME", name='cnn_out'),
             jax.nn.relu])

    def __call__(self, x_batch):
        return self.conv_model(x_batch)


if __name__ == '__main__':

    wandb.init(project="HiRo", name='CNN with gradient clipping')
    wandb.config = {
        'learning_rate': 0.01,
        'weight_decay': 0.0001,
        'epochs': 8,
        'batch_size': 8,
        'gradient_clip': 0.001,
        'cnn_layers': 5,
        'cnn_filters': 32,
        'dataset': 'dataset_3'
    }
    wandb.log(wandb.config)

    mongo = MongoClient(get_secret('mongo_connection_string'))

    train_imgs = download_dataset(mongo.hiro[wandb.config['dataset']], 'train')
    train_out = calculate_img_residual(train_imgs)
    train_in = train_imgs - train_out

    valid_imgs = download_dataset(mongo.hiro[wandb.config['dataset']], 'valid')
    valid_out = calculate_img_residual(valid_imgs)
    valid_in = valid_imgs - valid_out

    rng = jax.random.PRNGKey(220714)
    EPOCHS = wandb.config['epochs']
    BATCH_SIZE = wandb.config['batch_size']

    # pylint: disable=invalid-name
    def conv_net_fun(x):
        """
        Functional wrapper for cnn model
        :param x: model input
        """
        cnn = CNN(wandb.config)
        return cnn(x)

    conv_net = hk.transform(conv_net_fun)
    params = conv_net.init(rng, train_in[:2])
    optimizer = optax.adamw(learning_rate=wandb.config['learning_rate'],
                            weight_decay=wandb.config['weight_decay'])
    optimizer_state = optimizer.init(params)

    def mse(model_params, input_data, actual):
        """
        Functional wrapper for haiku model loss function
        :param model_params: model parameters
        :param input_data: input data for a forward pass
        :param actual: ground truth to calculate loss
        :return: mse loss
        """
        predicted = conv_net.apply(model_params, rng, input_data)
        return jnp.mean(actual - predicted) ** 2

    clip_limit = wandb.config['gradient_clip'] / wandb.config['learning_rate']
    batches = jnp.arange((train_in.shape[0] // BATCH_SIZE) + 1)
    for epoch in range(wandb.config['epochs']):

        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch * BATCH_SIZE), int(batch * BATCH_SIZE + BATCH_SIZE)
            else:
                start, end = int(batch * BATCH_SIZE), None

            loss, param_grads = jax.value_and_grad(mse)(params, train_in[start:end],
                                                        train_out[start:end])
            jax.tree_map(lambda array: jnp.minimum(jnp.maximum(-clip_limit, array), clip_limit),
                         param_grads)
            updates, optimizer_state = optimizer.update(param_grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            wandb.log({"gradient": param_grads}, step=epoch)

        train_loss = mse(params, train_in, train_out)
        valid_loss = mse(params, valid_in, valid_out)
        wandb.log({"train_loss": train_loss}, step=epoch)
        wandb.log({"valid_loss": valid_loss}, step=epoch)

        print(f'Epoch {epoch}, '
              f'train mse = {jnp.mean(train_loss)}, '
              f'valid mse = {jnp.mean(valid_loss)}')
