import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_network(hiddens, inpt, num_actions, scope='network', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for i, hidden in enumerate(hiddens):
            out = tf.layers.dense(out, hidden, name='d{}'.format(i),
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
            out = tf.nn.tanh(out)

        # policy branch
        mu = tf.layers.dense(
            out,
            num_actions,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-3e-3,
                maxval=3e-3
            ),
            name='mu'
        )
        mu = tf.nn.tanh(mu + 1e-5)

        sigma = tf.layers.dense(
            out,
            num_actions,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-3e-3,
                maxval=3e-3
            ),
            name='sigma'
        )
        sigma = tf.nn.softplus(sigma + 1e-5)

        dist = tf.distributions.Normal(mu, sigma)
        policy = tf.squeeze(dist.sample(num_actions), [0])

        # value branch
        value = tf.layers.dense(
            out,
            1,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-3e-3,
                maxval=3e-3
            ),
            name='d3'
        )
    return policy, value, dist

def make_network(hiddens):
    return lambda *args, **kwargs: _make_network(hiddens, *args, **kwargs)
