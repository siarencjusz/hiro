"""Script for training for local testing of the pipeline"""
import jax
from jax import numpy as jnp
import optax
import haiku as hk
from pymongo import MongoClient
import wandb
from utils import get_secret, download_dataset, calculate_img_residual, plot_sample


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
        'learning_rate': 0.1,
        'weight_decay': 0.0001,
        'epochs': 8,
        'batch_size': 8,
        'gradient_clip': 0.01,
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
    DEBUG_PLOT = True
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
        return jnp.mean((actual - predicted) ** 2)

    # if DEBUG_PLOT:
    #     plot_sample(train_in[8], train_out[8])

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
            # jax.tree_map(lambda array: jnp.minimum(jnp.maximum(-clip_limit, array), clip_limit),
            #              param_grads)

            updates, optimizer_state = optimizer.update(param_grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            wandb.log({"gradient": param_grads}, step=epoch)

        if DEBUG_PLOT:
            plot_predicted = conv_net.apply(params, rng, train_in)
            plot_sample(train_in[8], train_out[8], plot_predicted[8])

        train_mse = mse(params, train_in, train_out)
        valid_mse = mse(params, valid_in, valid_out)

        metrics = {'train_mse': train_mse,
                   'valid_mse': valid_mse,
                   'train_psnr': 10 * jnp.log10(1 / train_mse),
                   'valid_psnr': 10 * jnp.log10(1 / valid_mse)
                   }
        wandb.log(metrics, step=epoch)

        print(f'Epoch {epoch}, '
              f'train psnr = {jnp.round(metrics["train_psnr"], 2)}, '
              f'valid psnr = {jnp.round(metrics["valid_psnr"], 2)}')
