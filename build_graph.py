import numpy as np
import tensorflow as tf


def build_train(model,
                num_actions,
                optimizer,
                nenvs,
                lstm_unit=256,
                state_shape=[84, 84, 1],
                grad_clip=40.0,
                value_factor=0.5,
                entropy_factor=0.01,
                epsilon=0.2,
                scope='ppo',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # placeholers
        obs_input = tf.placeholder(tf.float32, [None] + state_shape, name='obs')
        rnn_state_ph0 = tf.placeholder(
            tf.float32, [nenvs, lstm_unit], name='rnn_state_0')
        rnn_state_ph1 = tf.placeholder(
            tf.float32, [nenvs, lstm_unit], name='rnn_state_1')
        actions_ph = tf.placeholder(tf.uint8, [None], name='action')
        target_values_ph = tf.placeholder(tf.float32, [None], name='value')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')
        step_size_ph = tf.placeholder(tf.int32, [], name='step_size')
        mask_ph = tf.placeholder(tf.bool, [None], name='mask')
        old_log_probs_ph = tf.placeholder(tf.float32, [None], name='old_log_prob')

        # rnn state in tuple
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            rnn_state_ph0, rnn_state_ph1)

        # network outpus
        policy, value, state_out = model(
            obs_input, rnn_state_tuple, num_actions,
            lstm_unit, nenvs, step_size_ph, scope='model')
        # network weights
        network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        log_prob = tf.reduce_sum(log_policy * actions_one_hot, axis=1, keep_dims=True)

        # loss
        advantages = tf.reshape(advantages_ph, [-1, 1])
        target_values = tf.reshape(target_values_ph, [-1, 1])
        with tf.variable_scope('value_loss'):
            masked_value_loss = tf.boolean_mask(
                tf.square(target_values - value), mask_ph)
            value_loss = tf.reduce_mean(masked_value_loss)
        with tf.variable_scope('entropy'):
            masked_entroypy = tf.boolean_mask(
                tf.reduce_sum(policy * log_policy, axis=1), mask_ph)
            entropy = -tf.reduce_mean(masked_entroypy)
        with tf.variable_scope('policy_loss'):
            old_log_prob = tf.reshape(old_log_probs_ph, [-1, 1])
            ratio = tf.exp(log_prob - old_log_prob)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(
                ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
            surr = tf.minimum(surr1, surr2)
            masked_policy_loss = tf.boolean_mask(surr, mask_ph)
            policy_loss = tf.reduce_mean(masked_policy_loss)
        loss = value_factor * value_loss - policy_loss - entropy_factor * entropy

        # gradients
        gradients = tf.gradients(loss, network_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
        # update
        grads_and_vars = zip(clipped_gradients, network_vars)
        optimize_expr = optimizer.apply_gradients(grads_and_vars)

        def train(obs, actions, targets, advantages, log_probs,
                  rnn_state0, rnn_state1, masks, step_size):
            feed_dict = {
                obs_input: obs,
                actions_ph: actions,
                target_values_ph: targets,
                advantages_ph: advantages,
                old_log_probs_ph: log_probs,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1,
                mask_ph: masks,
                step_size_ph: step_size
            }
            sess = tf.get_default_session()
            return sess.run([loss, optimize_expr], feed_dict=feed_dict)[0]

        def act(obs, rnn_state0, rnn_state1):
            feed_dict = {
                obs_input: obs,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1,
                step_size_ph: 1
            }
            sess = tf.get_default_session()
            return sess.run([policy, value, state_out], feed_dict=feed_dict)

    return act, train
