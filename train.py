"""Script for training for local testing of the pipeline"""
import os
from datetime import datetime
import jax
from jax import numpy as jnp
import optax
import haiku as hk
from pymongo import MongoClient
from tqdm import tqdm
import wandb
from constants import CNN_LAYERS, CNN_FILTERS, DATASET_NAME
from utils import get_secret, download_dataset, calculate_img_residual, plot_samples, jarray2json


class CNN(hk.Module):
    """Simple sequential convolutional neural network"""
    def __init__(self, arch):
        super().__init__(name="CNN")

        layers = []
        for _ in range(arch['cnn_layers'] - 1):
            layers.append(hk.Conv2D(output_channels=arch['cnn_filters'],
                                    kernel_shape=(3, 3), padding="SAME"))
            layers.append(jax.nn.relu)
        layers.append(hk.Conv2D(output_channels=3, kernel_shape=(3, 3),
                                padding="SAME", name='cnn_out'))
        self.conv_model = hk.Sequential(layers)

    def __call__(self, x_batch):
        return self.conv_model(x_batch)


if __name__ == '__main__':

    DEBUG_PLOT = False
    EXPERIMENT_NAME = 'CNN, 21pixels, clipping, L2'
    EXPERIMENT_TIMESTAMP = str(datetime.now())

    os.environ["WANDB_API_KEY"] = get_secret('wandb_api_key')
    os.environ["WANDB_MODE"] = "offline"

    wandb.init(project="HiRo", name=EXPERIMENT_NAME,
               settings=wandb.Settings(start_method="thread"))
    wandb.config = {
        'learning_rate': 0.1,
        'optimizer': 'sgd',
        'weight_decay': 0.0001,
        'sgd_momentum': 0.9,
        'epochs': 32,
        'batch_size': 64,
        'gradient_clip': 0.0001,
        'cnn_layers': CNN_LAYERS,
        'cnn_filters': CNN_FILTERS,
        'dataset': DATASET_NAME
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

    optimizer = optax.sgd(learning_rate=wandb.config['learning_rate'],
                          momentum=wandb.config['sgd_momentum'])
    optimizer_state = optimizer.init(params)

    def mse(model_params, input_data, actual, weight_decay=0.):
        """
        Functional wrapper for haiku model loss function
        :param model_params: model parameters
        :param input_data: input data for a forward pass
        :param actual: ground truth to calculate loss
        :param weight_decay: a weight for weight decay regularization
        :return: mse loss with a weight decay
        """
        all_params, _ = jax.tree_flatten(model_params)
        flat_params = jnp.concatenate([param.flatten() for param in all_params])
        predicted = conv_net.apply(model_params, rng, input_data)
        predicted_mse = jnp.mean(jnp.sum((actual - predicted) ** 2, axis=(1, 2, 3)))
        return predicted_mse + weight_decay * jnp.sum(flat_params ** 2)

    clip_limit = wandb.config['gradient_clip'] / wandb.config['learning_rate']
    sample_key = jax.random.PRNGKey(0)

    progress_bar = tqdm(range(wandb.config['epochs']), desc='Train', unit='epoch')
    for epoch in progress_bar:

        shuffle_id = jax.random.choice(sample_key, train_in.shape[0], [train_in.shape[0]],
                                       replace=False)
        sample_key = jax.random.PRNGKey(shuffle_id[0])
        train_in_shuffled = train_in[shuffle_id]
        train_out_shuffled = train_out[shuffle_id]

        for batch_start in range(0, train_in.shape[0], BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, train_in.shape[0])
            param_grads = jax.grad(mse)(params, train_in_shuffled[batch_start:batch_end],
                                        train_out_shuffled[batch_start:batch_end],
                                        wandb.config['weight_decay'])
            param_grads = jax.tree_map(lambda x: jnp.maximum(x, -clip_limit), param_grads)
            param_grads = jax.tree_map(lambda x: jnp.minimum(x, clip_limit), param_grads)
            updates, optimizer_state = optimizer.update(param_grads, optimizer_state)
            params = optax.apply_updates(params, updates)

        if DEBUG_PLOT:
            plot_train_idx = jnp.array([8, 12])
            plot_valid_idx = jnp.array([0, 1, 4, 5])
            plot_train_predicted = conv_net.apply(params, rng, train_in[plot_train_idx])
            plot_valid_predicted = conv_net.apply(params, rng, valid_in[plot_valid_idx])
            plot_samples(
                jnp.concatenate((train_in[plot_train_idx], valid_in[plot_valid_idx]), axis=0),
                jnp.concatenate((train_out[plot_train_idx], valid_out[plot_valid_idx]), axis=0),
                jnp.concatenate((plot_train_predicted, plot_valid_predicted), axis=0))

        train_mse = float(mse(params, train_in, train_out))
        valid_mse = float(mse(params, valid_in, valid_out))

        metrics = {'train_mse': train_mse,
                   'valid_mse': valid_mse,
                   'train_psnr': 10 * jnp.log10((wandb.config['cnn_layers'] * 2 + 1) ** 2 /
                                                train_mse),
                   'valid_psnr': 10 * jnp.log10((wandb.config['cnn_layers'] * 2 + 1) ** 2 /
                                                valid_mse)}
        wandb.log(metrics, step=epoch)
        mongo.hiro.models.insert_one({'training_timestamp': EXPERIMENT_TIMESTAMP,
                                      'experiment_name': EXPERIMENT_NAME,
                                      'epoch': epoch + 1,
                                      'metrics': metrics,
                                      'architecture': wandb.config,
                                      'params': jax.tree_map(jarray2json, params)})
        progress_bar.set_description(f'Train psnr = {metrics["train_psnr"]:.2f}dB, '
                                     f'valid psnr = {metrics["valid_psnr"]:.2f}dB')
