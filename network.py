import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_network(convs,
                  fcs,
                  lstm,
                  padding,
                  inpt,
                  rnn_state_tuple,
                  num_actions,
                  lstm_unit,
                  nenvs,
                  step_size,
                  scope,
                  reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('convnet'):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(
                    out,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.orthogonal_initializer(np.sqrt(2.0))
                )
            out = layers.flatten(out)

        with tf.variable_scope('hiddens'):
            for hidden in fcs:
                out = layers.fully_connected(
                    out, hidden, activation_fn=tf.nn.relu,
                    weights_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_unit, state_is_tuple=True)
            # sequence to batch
            rnn_in = tf.reshape(out, [nenvs, step_size, int(out.shape[1])])
            sequence_length = tf.ones(nenvs, dtype=tf.int32) * step_size
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=rnn_state_tuple,
                sequence_length=sequence_length, time_major=False)
            # batch to sequence
            rnn_out = tf.reshape(lstm_outputs, [-1, lstm_unit])

        if lstm:
            out = rnn_out

        policy = layers.fully_connected(
            out, num_actions, activation_fn=tf.nn.softmax,
            weights_initializer=tf.orthogonal_initializer(0.1))

        value = layers.fully_connected(
            out, 1, activation_fn=None,
            weights_initializer=tf.orthogonal_initializer(1.0))

    return policy, value, (lstm_state[0], lstm_state[1])

def make_network(convs, fcs, lstm=True, padding='VALID'):
    return lambda *args, **kwargs: _make_network(convs, fcs, lstm, padding, *args, **kwargs)
