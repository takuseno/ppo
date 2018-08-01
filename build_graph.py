import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(network, obs_dim,
            num_actions, gamma=1.0, epsilon=0.2, beta=0.01, scope='ppo', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # input placeholders
        obs_t_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_t')
        act_t_ph = tf.placeholder(tf.float32, [None, num_actions], name='action')
        return_t_ph = tf.placeholder(tf.float32, [None, 1], name='return')
        advantage_t_ph = tf.placeholder(tf.float32, [None, 1], name='advantage')

        policy, value, dist = network(
            obs_t_input,
            num_actions,
            scope='network',
            reuse=reuse
        )
        network_func_vars = util.scope_vars(
            util.absolute_scope_name('network'),
            trainable_only=True
        )

        old_policy, old_value, old_dist = network(
            obs_t_input,
            num_actions,
            scope='old_network',
            reuse=reuse
        )
        old_network_func_vars = util.scope_vars(
            util.absolute_scope_name('old_network'),
            trainable_only=True
        )

        tmp_policy, tmp_value, tmp_dist = network(
            obs_t_input,
            num_actions,
            scope='tmp_network',
            reuse=reuse
        )
        tmp_network_func_vars = util.scope_vars(
            util.absolute_scope_name('tmp_network'),
            trainable_only=True
        )

        # clipped surrogate objective
        cur_policy = dist.log_prob(act_t_ph + 1e-5)
        old_policy = old_dist.log_prob(act_t_ph + 1e-5)
        ratio = tf.exp(cur_policy - old_policy)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)
        surrogate = -tf.reduce_mean(
            tf.minimum(ratio * advantage_t_ph, clipped_ratio * advantage_t_ph),
            name='surrogate')

        with tf.variable_scope('loss'):
            # value network loss
            value_loss = tf.reduce_mean(tf.square(value - return_t_ph))

            # entropy penalty for exploration
            entropy = tf.reduce_mean(dist.entropy())
            penalty = -beta * entropy

            # total loss
            loss = surrogate + value_loss + penalty

        # optimize operations
        optimizer = tf.train.AdamOptimizer(3 * 1e-4)
        optimize_expr = optimizer.minimize(loss, var_list=network_func_vars)

        # update old network operations
        with tf.variable_scope('update_old_network'):
            update_old_expr = []
            sorted_tmp_vars = sorted(
                tmp_network_func_vars,
                key=lambda v: v.name
            )
            sorted_old_vars = sorted(
                old_network_func_vars,
                key=lambda v: v.name
            )
            for var_tmp, var_old in zip(sorted_tmp_vars, sorted_old_vars):
                update_old_expr.append(var_old.assign(var_tmp))
            update_old_expr = tf.group(*update_old_expr)

        # update tmp network operations
        with tf.variable_scope('update_tmp_network'):
            update_tmp_expr = []
            sorted_vars = sorted(network_func_vars, key=lambda v: v.name)
            sorted_tmp_vars = sorted(
                tmp_network_func_vars,
                key=lambda v: v.name
            )
            for var, var_tmp in zip(sorted_vars, sorted_tmp_vars):
                update_tmp_expr.append(var_tmp.assign(var))
            update_tmp_expr = tf.group(*update_tmp_expr)

        # action theano-style function
        act = util.function(inputs=[obs_t_input], outputs=[policy, value])

        # train theano-style function
        train = util.function(
            inputs=[
                obs_t_input, act_t_ph, return_t_ph, advantage_t_ph
            ],
            outputs=[loss, value_loss, tf.reduce_mean(ratio)],
            updates=[optimize_expr]
        )

        # update target theano-style function
        update_old = util.function([], [], updates=[update_old_expr])
        backup_current = util.function([], [], updates=[update_tmp_expr])

        return act, train, update_old, backup_current
