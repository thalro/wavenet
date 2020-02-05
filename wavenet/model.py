from tensorflow.keras.layers import Activation, Add, Conv1D, Dense, Flatten, Input, Multiply
from tensorflow.keras.models import Model


def WaveNetResidualConv1D(dilation_channels,skip_channels,residual_channels, kernel_size, dilation_rate):
    """ Function that creates a residual block for the WaveNet with gated
        activation units, skip connections and residual output, as described
        in Sections 2.3 and 2.4 of the paper [1].

        Args:
            num_filters (int): Number of filters used for convolution.
            kernel_size (int): The size of the convolution.
            dilation_rate (int): The dilation rate for the dilated convolution.

        Returns:
            A layer wrapper compatible to the Keras functional API.

        See:
            [1] Oord, Aaron van den, et al. "Wavenet: A generative model for
                raw audio." arXiv preprint arXiv:1609.03499 (2016).
    """
    def build_residual_block(l_input):
        # Gated activation.
        l_sigmoid_conv1d = Conv1D(
            dilation_channels, kernel_size, dilation_rate=dilation_rate,
            padding="causal", activation="sigmoid")(l_input)
        l_tanh_conv1d = Conv1D(
            dilation_channels, kernel_size, dilation_rate=dilation_rate,
            padding="causal", activation="tanh")(l_input)
        l_mul = Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
        # Branches out to skip unit and residual output.
        l_skip_connection = Conv1D(skip_channels, 1)(l_mul)
        
        l_residual = Add()([l_input, Conv1D(residual_channels,1)(l_mul)])
        return l_residual, l_skip_connection
    
    return build_residual_block


def build_wavenet_model(num_stacks, dilation_channels=32,
                        skip_channels=64,
                        residual_channels=32,
                        num_layers_per_stack = 9,
                        input_channels = 1):
    """ Returns an implementation of WaveNet, as described in Section 2
        of the paper [1].

        Args:
            num_stacks: number of stacks of dilated convolutions
            num_layers_per_stack: number of dilated convolutions per stack
            num_filters (int): Number of filters used for convolution.
            num_residual_blocks (int): How many residual blocks to generate
                between input and output. Residual block i will have a dilation
                rate of 2^(i+1), i starting from zero.

        Returns:
            A Keras model representing the WaveNet., the recetive field size

        See:
            [1] Oord, Aaron van den, et al. "Wavenet: A generative model for
                raw audio." arXiv preprint arXiv:1609.03499 (2016).
    """
    kernel_size = 2
    receptive_field_size = num_stacks*2**(num_layers_per_stack+1)
    l_input = Input(batch_shape=(None, receptive_field_size, input_channels))
    l_stack_conv1d = Conv1D(residual_channels, kernel_size, padding="causal")(l_input)
    l_skip_connections = []
    for i in range(num_stacks*num_layers_per_stack+num_stacks-1):
        dilution = 2 ** ((i + 1)%(num_layers_per_stack+1))
        l_stack_conv1d, l_skip_connection = WaveNetResidualConv1D(
            dilation_channels,skip_channels,residual_channels, kernel_size, dilution)(l_stack_conv1d)
        l_skip_connections.append(l_skip_connection)
    if len(l_skip_connections)>1:
    	l_sum = Add()(l_skip_connections)
    else:
        l_sum = l_skip_connections[0]

    relu = Activation("relu")(l_sum)
    l1_conv1d = Conv1D(skip_channels, 1, activation="relu")(relu)
    l2_conv1d = Conv1D(skip_channels, 1)(l1_conv1d)
    l_flatten = Flatten()(l2_conv1d)
    l_output = Dense(256, activation="softmax")(l_flatten)
    model = Model(inputs=[l_input], outputs=[l_output])
    
    return model,receptive_field_size
