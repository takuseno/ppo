import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from rlsaber.tf_util import lstm, batch_to_seq, seq_to_batch


def make_cnn(convs, padding, inpt, initializer=None):
    if initializer is None:
        initializer = tf.orthogonal_initializer(np.sqrt(2.0))
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
                weights_initializer=initializer
            )
    return out

def make_fcs(fcs, inpt, activation=tf.nn.relu, initializer=None):
    if initializer is None:
        initializer = tf.orthogonal_initializer(np.sqrt(2.0))
    out = inpt
    with tf.variable_scope('hiddens'):
        for hidden in fcs:
            out = layers.fully_connected(out, hidden, activation_fn=activation,
                                         weights_initializer=initializer)
    return out

def make_lstm(lstm_unit, nenvs, step_size, inpt, masks, rnn_state):
    with tf.variable_scope('rnn'):
        rnn_in = batch_to_seq(inpt, nenvs, step_size)
        masks = batch_to_seq(masks, nenvs, step_size)
        rnn_out, rnn_state = lstm(
            rnn_in, masks, rnn_state, lstm_unit, np.sqrt(2.0))
        rnn_out = seq_to_batch(rnn_out, nenvs, step_size)
    return rnn_out, rnn_state

def cnn_network(convs,
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
    out = make_cnn(convs, padding, inpt)
    out = layers.flatten(out)
    out = make_fcs(fcs, out)
    rnn_out, rnn_state = make_lstm(
        lstm_unit, nenvs, step_size, out, masks, rnn_state)

    if use_lstm:
        out = rnn_out

    policy = layers.fully_connected(
        out, num_actions, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(0.1))
    dist = tf.distributions.Categorical(probs=tf.nn.softmax(policy))

    value = layers.fully_connected(
        out, 1, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(1.0))

    return dist, value, rnn_state

def mlp_network(fcs,
                use_lstm,
                inpt,
                masks,
                rnn_state,
                num_actions,
                lstm_unit,
                nenvs,
                step_size,
                scope):
    policy_rnn_state, value_rnn_state = tf.split(rnn_state, 2, axis=-1)

    inpt = layers.flatten(inpt)
    input_dim = inpt.get_shape().as_list()[1] + 1
    def initializer(scale):
        return tf.random_normal_initializer(stddev=np.sqrt(scale / input_dim))

    with tf.variable_scope('policy'):
        out = make_fcs(
            fcs, inpt, activation=tf.nn.tanh, initializer=initializer(1.0))
        rnn_out, policy_rnn_state = make_lstm(
            lstm_unit//2, nenvs, step_size, out, masks, policy_rnn_state)

        if use_lstm:
            out = rnn_out

        policy = layers.fully_connected(out, num_actions, activation_fn=None,
                                        weights_initializer=initializer(0.01))
        logstd = tf.get_variable(name='logstd', shape=[1, num_actions],
                                 initializer=tf.zeros_initializer())
        std = tf.zeros_like(policy) + tf.exp(logstd)
        dist = tf.distributions.Normal(loc=policy, scale=std)

    with tf.variable_scope('value'):
        out = make_fcs(
            fcs, inpt, activation=tf.nn.tanh, initializer=initializer(1.0))
        rnn_out, value_rnn_state = make_lstm(
            lstm_unit//2, nenvs, step_size, out, masks, value_rnn_state)

        if use_lstm:
            out = rnn_out

        value = layers.fully_connected(
            out, 1, activation_fn=None, weights_initializer=initializer(1.0))

    rnn_state = tf.concat([policy_rnn_state, value_rnn_state], axis=-1)

    return dist, value, rnn_state


def _make_network(convs,
                  fcs,
                  use_lstm,
                  padding,
                  continuous,
                  inpt,
                  masks,
                  rnn_state,
                  num_actions,
                  lstm_unit,
                  nenvs,
                  step_size,
                  scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if continuous:
            return mlp_network(fcs, use_lstm, inpt, masks, rnn_state,
                               num_actions, lstm_unit, nenvs, step_size, scope)
        else:
            return cnn_network(convs, fcs, use_lstm, padding, inpt, masks,
                               rnn_state, num_actions, lstm_unit, nenvs,
                               step_size, scope)

def make_network(convs, fcs, use_lstm=True, padding='VALID', continuous=False):
    return lambda *args, **kwargs: _make_network(convs, fcs, use_lstm, padding,\
        continuous, *args, **kwargs)
