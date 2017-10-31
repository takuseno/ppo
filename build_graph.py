import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(network, obs_dim, num_actions, gamma=1.0, epsilon=0.2, scope='ppo', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # input placeholders
        obs_t_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_t')
        act_t_ph = tf.placeholder(tf.float32, [None, num_actions], name='action')
        return_t_ph = tf.placeholder(tf.float32, [None], name='return')
        advantage_t_ph = tf.placeholder(tf.float32, [None], name='advantage')
        lr_ph = tf.placeholder(tf.float32, [None], name='learning_rate')

        policy, value, dist = network(obs_t_input, num_actions, scope='network', reuse=reuse)
        network_func_vars = util.scope_vars(util.absolute_scope_name('network'), trainable_only=True)

        old_policy, old_value, old_dist = network(obs_t_input, num_actions, scope='old_network', reuse=reuse)
        old_network_func_vars = util.scope_vars(util.absolute_scope_name('old_network'), trainable_only=True)

        # clipped surrogate objective
        ratio = tf.exp(dist.log_prob(act_t_ph) - old_dist.log_prob(act_t_ph))
        surrogate1 = ratio * advantage_t_ph
        surrogate2 = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage_t_ph
        surrogate = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2), name='surrogate')

        value_loss = tf.reduce_mean(tf.square(value - returns))

        entropy = tf.reduce_mean(dist.entropy())
        penalty = -0.01 * entropy

        # total loss
        loss = surrogate + value_loss + penalty

        # optimize operations
        optimizer = tf.train.AdamOptimizer(lr_ph)
        optimize_expr = optimizer.minimize(loss, var_list=network_func_vars)

        # update old network operations
        with tf.variable_scope('update_old_network'):
            update_old_expr = []
            for var, var_old in zip(sorted(network_func_vars, key=lambda v: v.name),
                                        sorted(old_network_func_vars, key=lambda v: v.name)):
                update_old_expr.append(var_old.assign(var))
            update_old_expr = tf.group(*update_old_expr)

        # action theano-style function
        act = util.function(inputs=[obs_t_input], outputs=[policy, value])

        # train theano-style function
        train = util.function(
            inputs=[
                obs_t_input, act_t_ph, return_t_ph, advantage_t_ph, lr_ph
            ],
            outputs=[loss],
            updates=[optimize_expr]
        )

        # update target theano-style function
        update_old = util.function([], [], updates=[update_old_expr])

        return act, train, update_old
