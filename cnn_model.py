"""Sequential convolutional neural network model with all auxiliary utilities"""
import haiku as hk
import jax


class CNN(hk.Module):
    """Simple sequential convolutional neural network"""
    def __init__(self, model_hyper_params):
        super().__init__(name="CNN")

        layers = []
        for _ in range(model_hyper_params['cnn_layers'] - 1):
            layers.append(hk.Conv2D(output_channels=model_hyper_params['cnn_filters'],
                                    kernel_shape=(3, 3), padding="SAME"))
            layers.append(jax.nn.relu)
        layers.append(hk.Conv2D(output_channels=3, kernel_shape=(3, 3),
                                padding="SAME", name='cnn_out'))
        self.conv_model = hk.Sequential(layers)

    def __call__(self, x_batch):
        return self.conv_model(x_batch)
