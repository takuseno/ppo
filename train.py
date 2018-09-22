import threading
import multiprocessing
import argparse
import cv2
import gym
import copy
import os
import time
import atari_constants
import box_constants
import numpy as np
import tensorflow as tf

from rlsaber.log import TfBoardLogger, dump_constants
from rlsaber.trainer import BatchTrainer
from rlsaber.env import EnvWrapper, BatchEnvWrapper, NoopResetEnv, EpisodicLifeEnv, MaxAndSkipEnv
from rlsaber.preprocess import atari_preprocess
from network import make_network
from agent import Agent
from datetime import datetime


def main():
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--load', type=str)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.logdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    env_name = args.env
    tmp_env = gym.make(env_name)
    is_atari = len(tmp_env.observation_space.shape) != 1
    if not is_atari:
        observation_space = tmp_env.observation_space
        constants = box_constants
        actions = range(tmp_env.action_space.n)
        state_shape = [observation_space.shape[0], constants.STATE_WINDOW]
        state_preprocess = lambda s: s
        # (window_size, dim) -> (dim, window_size)
        phi = lambda s: np.transpose(s, [1, 0])
    else:
        constants = atari_constants
        actions = range(tmp_env.action_space.n)
        state_shape = constants.STATE_SHAPE + [constants.STATE_WINDOW]
        def state_preprocess(state):
            state = atari_preprocess(state, constants.STATE_SHAPE)
            state = np.array(state, dtype=np.float32)
            return state / 255.0
        # (window_size, H, W) -> (H, W, window_size)
        phi = lambda s: np.transpose(s, [1, 2, 0])

    # save settings
    dump_constants(constants, os.path.join(outdir, 'constants.json'))

    sess = tf.Session()
    sess.__enter__()

    model = make_network(
        constants.CONVS, constants.FCS,
        use_lstm=constants.LSTM, padding=constants.PADDING)

    # learning rate with decay operation
    lr = tf.Variable(constants.LR)
    decayed_lr = tf.placeholder(tf.float32)
    decay_lr_op = lr.assign(decayed_lr)
    optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-5)
    # epsilon with decay operation
    epsilon = tf.Variable(constants.EPSILON)
    decayed_epsilon = tf.placeholder(tf.float32)
    decay_epsilon_op = epsilon.assign(decayed_epsilon)

    agent = Agent(
        model,
        actions,
        optimizer,
        nenvs=constants.ACTORS,
        gamma=constants.GAMMA,
        lam=constants.LAM,
        lstm_unit=constants.LSTM_UNIT,
        value_factor=constants.VALUE_FACTOR,
        entropy_factor=constants.ENTROPY_FACTOR,
        time_horizon=constants.TIME_HORIZON,
        batch_size=constants.BATCH_SIZE,
        grad_clip=constants.GRAD_CLIP,
        epsilon=epsilon,
        state_shape=state_shape,
        epoch=constants.EPOCH,
        phi=phi
    )

    saver = tf.train.Saver()
    if args.load:
        saver.restore(sess, args.load)

    # create environemtns
    envs = []
    for i in range(constants.ACTORS):
        env = gym.make(args.env)
        env.seed(constants.RANDOM_SEED)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env)
            env = EpisodicLifeEnv(env)
        wrapped_env = EnvWrapper(
            env,
            r_preprocess=lambda r: np.clip(r, -1.0, 1.0),
            s_preprocess=state_preprocess
        ) 
        envs.append(wrapped_env)
    batch_env = BatchEnvWrapper(envs)

    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    logger = TfBoardLogger(summary_writer)
    logger.register('reward', dtype=tf.float32)
    end_episode = lambda r, s, e: logger.plot('reward', r, s)

    def after_action(state, reward, global_step, local_step):
        if constants.LR_DECAY == 'linear':
            decay = 1.0 - (float(global_step) / constants.FINAL_STEP)
            if decay < 0.0:
                decay = 0.0
            sess.run(decay_lr_op, feed_dict={decayed_lr: constants.LR * decay})
            sess.run(decay_epsilon_op,
                     feed_dict={decayed_epsilon: constants.EPSILON * decay})
        if global_step % 10 ** 6 == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=global_step)

    trainer = BatchTrainer(
        env=batch_env,
        agent=agent,
        render=args.render,
        state_shape=state_shape[:-1],
        state_window=constants.STATE_WINDOW,
        time_horizon=constants.TIME_HORIZON,
        batch_size=constants.BATCH_SIZE,
        final_step=constants.FINAL_STEP,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo
    )
    trainer.start()

if __name__ == '__main__':
    main()
