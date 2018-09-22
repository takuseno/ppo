import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from rlsaber.tf_util import lstm, batch_to_seq, seq_to_batch


def _make_network(convs,
                  fcs,
                  use_lstm,
                  padding,
                  inpt,
                  masks,
                  rnn_state,
                  num_actions,
                  lstm_unit,
                  nenvs,
                  step_size,
                  scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
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
            rnn_in = batch_to_seq(out, nenvs, step_size)
            masks = batch_to_seq(masks, nenvs, step_size)
            rnn_out, rnn_state = lstm(
                rnn_in, masks, rnn_state, lstm_unit, np.sqrt(2.0))
            rnn_out = seq_to_batch(rnn_out, nenvs, step_size)

        if use_lstm:
            out = rnn_out

        policy = layers.fully_connected(
            out, num_actions, activation_fn=tf.nn.softmax,
            weights_initializer=tf.orthogonal_initializer(0.1))

        value = layers.fully_connected(
            out, 1, activation_fn=None,
            weights_initializer=tf.orthogonal_initializer(1.0))

    return policy, value, rnn_state

def make_network(convs, fcs, use_lstm=True, padding='VALID'):
    return lambda *args, **kwargs: _make_network(convs, fcs, use_lstm, padding, *args, **kwargs)
