import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

# TODO: uncomment
__all__ = [
    'RNN',
    'SimpleRNN',
    'GRURNN',
    'LSTMRNN',
    'BiRNN',
    # 'ConvRNNCell',
    # 'BasicConvLSTMCell',
    # 'ConvLSTM',
    'retrieve_seq_length_op',
    'retrieve_seq_length_op2',
    'retrieve_seq_length_op3',
    'target_mask_op',
]


class RNN(Layer):
    """
    ###########################################################################
    This is the modified RNN desiginated for the Text Antispam project.
    We did a tiny modification on RNN.forward to achieve dynamic RNN behaviour.

    This modification is not an official fix for the dynamic RNN problem.
    ###########################################################################

    The :class:`RNN` class is a fixed length recurrent layer for implementing simple RNN,
    LSTM, GRU and etc.

    Parameters
    ----------
    cell : TensorFlow cell function
        A RNN cell implemented by tf.keras
            - E.g. tf.keras.layers.SimpleRNNCell, tf.keras.layers.LSTMCell, tf.keras.layers.GRUCell
            - Note TF2.0+, TF1.0+ and TF1.0- are different

    return_last_output : boolean
        Whether return last output or all outputs in a sequence.

            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False

        In a dynamic model, `return_last_output` can be updated when it is called in customised forward().
        By default, `False`.
    return_seq_2d : boolean
        Only consider this argument when `return_last_output` is `False`

            - If True, return 2D Tensor [batch_size * n_steps, n_hidden], for stacking Dense layer after it.
            - If False, return 3D Tensor [batch_size, n_steps, n_hidden], for stacking multiple RNN after it.

        In a dynamic model, `return_seq_2d` can be updated when it is called in customised forward().
        By default, `False`.
    return_last_state: boolean
        Whether to return the last state of the RNN cell. The state is a list of Tensor.
        For simple RNN and GRU, last_state = [last_output]; For LSTM, last_state = [last_output, last_cell_state]

            - If True, the layer will return outputs and the final state of the cell.
            - If False, the layer will return outputs only.

        In a dynamic model, `return_last_state` can be updated when it is called in customised forward().
        By default, `False`.
    in_channels: int
        Optional, the number of channels of the previous layer which is normally the size of embedding.
        If given, the layer will be built when init.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : str
        A unique layer name.

    Examples
    --------
    For synced sequence input and output, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm.py>`__

    A simple regression model below.

    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> rnn_out, lstm_state = tl.layers.RNN(
    >>>     cell=tf.keras.layers.LSTMCell(units=hidden_size, dropout=0.1),
    >>>     in_channels=embedding_size,
    >>>     return_last_output=True, return_last_state=True, name='lstmrnn'
    >>> )(inputs)
    >>> outputs = tl.layers.Dense(n_units=1)(rnn_out)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_state[0], rnn_state[1]], name='rnn_model')
    >>> # If LSTMCell is applied, the rnn_state is [h, c] where h the hidden state and c the cell state of LSTM.

    A stacked RNN model.

    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> rnn_out1 = tl.layers.RNN(
    >>>     cell=tf.keras.layers.SimpleRNNCell(units=hidden_size, dropout=0.1),
    >>>     return_last_output=False, return_seq_2d=False, return_last_state=False
    >>> )(inputs)
    >>> rnn_out2 = tl.layers.RNN(
    >>>     cell=tf.keras.layers.SimpleRNNCell(units=hidden_size, dropout=0.1),
    >>>     return_last_output=True, return_last_state=False
    >>> )(rnn_out1)
    >>> outputs = tl.layers.Dense(n_units=1)(rnn_out2)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=outputs)

    An example if the sequences have different length and contain padding.
    Similar to the DynamicRNN in TL 1.x.

    If the `sequence_length` is provided in RNN's forwarding and both `return_last_output` and `return_last_state`
    are set as `True`, the forward function will automatically ignore the paddings. Note that if `return_last_output`
    is set as `False`, the synced sequence outputs will still include outputs which correspond with paddings,
    but users are free to select which slice of outputs to be used in following procedure.

    The `sequence_length` should be a list of integers which indicates the length of each sequence.
    It is recommended to
    `tl.layers.retrieve_seq_length_op3 <https://tensorlayer.readthedocs.io/en/latest/modules/layers.html#compute-sequence-length-3>`__
    to calculate the `sequence_length`.

    >>> data = [[[1], [2], [0], [0], [0]], [[1], [2], [3], [0], [0]], [[1], [2], [6], [1], [1]]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> class DynamicRNNExample(tl.models.Model):
    >>>     def __init__(self):
    >>>         super(DynamicRNNExample, self).__init__()
    >>>         self.rnnlayer = tl.layers.RNN(
    >>>             cell=tf.keras.layers.SimpleRNNCell(units=6, dropout=0.1), in_channels=1, return_last_output=True,
    >>>             return_last_state=True
    >>>         )
    >>>     def forward(self, x):
    >>>         z, s = self.rnnlayer(x, sequence_length=tl.layers.retrieve_seq_length_op3(x))
    >>>         return z, s
    >>> model = DynamicRNNExample()
    >>> model.eval()
    >>> output, state = model(data)


    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see layer :class:`Reshape`.

    """

    def __init__(
            self,
            cell,
            return_last_output=False,
            return_seq_2d=False,
            return_last_state=True,
            in_channels=None,
            name=None,  # 'rnn'
    ):

        super(RNN, self).__init__(name=name)

        self.cell = cell
        self.return_last_output = return_last_output
        self.return_seq_2d = return_seq_2d
        self.return_last_state = return_last_state

        if in_channels is not None:
            self.build((None, None, in_channels))
            self._built = True

        logging.info("RNN %s: cell: %s, n_units: %s" % (self.name, self.cell.__class__.__name__, self.cell.units))

    def __repr__(self):
        s = ('{classname}(cell={cellname}, n_units={n_units}')
        s += ', name=\'{name}\''
        s += ')'
        return s.format(
            classname=self.__class__.__name__, cellname=self.cell.__class__.__name__, n_units=self.cell.units,
            **self.__dict__
        )

    def build(self, inputs_shape):
        """
        Parameters
        ----------
        inputs_shape : tuple
            the shape of inputs tensor
        """
        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        if len(inputs_shape) != 3:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        with tf.name_scope(self.name) as scope:
            self.cell.build(tuple(inputs_shape))

        if self._trainable_weights is None:
            self._trainable_weights = list()
        for var in self.cell.trainable_variables:
            self._trainable_weights.append(var)

    # @tf.function
    def forward(self, inputs, sequence_length=None, initial_state=None, **kwargs):
        """
        Parameters
        ----------
        inputs : input tensor
            The input of a network
        sequence_length: None or list of integers
            The actual length of each sequence in batch without padding.
            If provided, when `return_last_output` and `return_last_state` are `True`,
            the RNN will perform in the manner of a dynamic RNN, i.e.
            the RNN will return the actual last output / state without padding.
        initial_state : None or list of Tensor (RNN State)
            If None, `initial_state` is zero state.

        **kwargs: dict
            Some attributes can be updated during forwarding
            such as `return_last_output`, `return_seq_2d`, `return_last_state`.
        """
        ############# modificatification #############
        # compute sequence length inside RNN.forward to avoid missing this argument
        sequence_length = tl.layers.retrieve_seq_length_op3(inputs)
        # print(sequence_length)
        ##############################################

        if kwargs:
            for attr in kwargs:
                if attr in self.__dict__:
                    setattr(self, attr, kwargs[attr])

        batch_size = inputs.get_shape().as_list()[0]
        total_steps = inputs.get_shape().as_list()[1]

        # checking the type and values of sequence_length
        if sequence_length is not None:
            if isinstance(sequence_length, list):
                pass
            elif isinstance(sequence_length, tf.Tensor):
                pass
            elif isinstance(sequence_length, np.ndarray):
                sequence_length = sequence_length.tolist()
            else:
                raise TypeError(
                    "The argument sequence_length should be either None or a list of integers. "
                    "Type got %s" % type(sequence_length)
                )
            if (len(sequence_length) != batch_size):
                raise ValueError(
                    "The argument sequence_length should contain %d " % batch_size +
                    "elements indicating the initial length of each sequence, but got only %d. " % len(sequence_length)
                )
            for i in sequence_length:
                if not (type(i) is int or (isinstance(i, tf.Tensor) and i.dtype.is_integer)):
                    raise TypeError(
                        "The argument sequence_length should be either None or a list of integers. "
                        "One element of sequence_length has the type %s" % type(i)
                    )
                if i > total_steps:
                    raise ValueError(
                        "The actual length of a sequence should not be longer than "
                        "that of the longest sequence (total steps) in this mini-batch. "
                        "Total steps of this mini-batch %d, " % total_steps +
                        "but got an actual length of a sequence %d" % i
                    )

            sequence_length = [i - 1 if i >= 1 else 0 for i in sequence_length]

        # set warning
        # if (not self.return_last_output) and sequence_length is not None:
        #     warnings.warn(
        #         'return_last_output is set as %s ' % self.return_last_output +
        #         'When sequence_length is provided, it is recommended to set as True. ' +
        #         'Otherwise, padding will be considered while RNN is forwarding.'
        #     )

        # return the last output, iterating each seq including padding ones. No need to store output during each
        # time step.
        if self.return_last_output and sequence_length is None:
            outputs = [-1]
        else:
            outputs = list()

        # initialize the states if provided
        states = initial_state if initial_state is not None else self.cell.get_initial_state(inputs)
        if not isinstance(states, list):
            states = [states]

        stored_states = list()

        # initialize the cell
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()

        # recurrent computation
        # FIXME: if sequence_length is provided (dynamic rnn), only iterate max(sequence_length) times.
        for time_step in range(total_steps):

            cell_output, states = self.cell.call(inputs[:, time_step, :], states, training=self.is_train)
            stored_states.append(states)

            if self.return_last_output and sequence_length is None:
                outputs[-1] = cell_output
            else:
                outputs.append(cell_output)

        # prepare to return results
        if self.return_last_output and sequence_length is None:
            outputs = outputs[-1]

        elif self.return_last_output and sequence_length is not None:
            outputs = tf.convert_to_tensor(outputs)
            outputs = tf.gather(outputs, sequence_length, axis=0)

            outputs_without_padding = []
            for i in range(batch_size):
                outputs_without_padding.append(outputs[i][i][:])
            outputs = tf.convert_to_tensor(outputs_without_padding)
        else:
            if self.return_seq_2d:
                # PTB tutorial: stack dense layer after that, or compute the cost from the output
                # 2D Tensor [batch_size * n_steps, n_hidden]
                outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.cell.units])
            else:
                # <akara>: stack more RNN layer after that
                # 3D Tensor [batch_size, n_steps, n_hidden]
                outputs = tf.reshape(tf.concat(outputs, 1), [-1, total_steps, self.cell.units])

        if self.return_last_state and sequence_length is None:
            return outputs, states
        elif self.return_last_state and sequence_length is not None:

            stored_states = tf.convert_to_tensor(stored_states)
            stored_states = tf.gather(stored_states, sequence_length, axis=0)

            states = []
            for i in range(stored_states.shape[1]):
                states.append(tf.convert_to_tensor([stored_states[b, i, b, :] for b in range(batch_size)]))

            return outputs, states
        else:
            return outputs