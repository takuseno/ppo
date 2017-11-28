import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.rl.replay_buffer import ReplayBuffer
from network import make_network
from agent import Agent
from backup import Backup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-steps', type=int, default=10 ** 7)
    parser.add_argument('--actors', type=int, default=8)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
    if args.logdir is None:
        args.logdir = os.path.join(os.path.dirname(__file__), 'logs')

    envs = []
    for i in range(args.actors):
        envs.append(gym.make(args.env))

    obs_dim = envs[0].observation_space.shape[0]
    n_actions = envs[0].action_space.shape[0]

    network = make_network([64, 64])

    sess = tf.Session()
    sess.__enter__()

    agent = Agent(network, obs_dim, n_actions)

    initialize()
    agent.sync_old()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    reward_summary = tf.placeholder(tf.int32, (), name='reward_summary')
    tf.summary.scalar('reward_summary', reward_summary)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.logdir, sess.graph)

    global_step = 0
    episode = 0
    backup = Backup(args.actors)
    while True:
        training_data = []
        for i in range(args.actors):
            env = envs[i]
            # restore previous situation
            sum_of_reward, reward, obs, last_obs,\
                    last_action, last_value, done = backup.restore(i)
            # initialize values for the new episode
            if done:
                sum_of_reward = 0
                reward = 0
                obs = env.reset()
                last_obs = None
                last_action = None
                last_value = None
                done = False
            for step in range(100):
                if i == 0 and args.render:
                    env.render()

                action, value = agent.act_and_train(
                        last_obs, last_action, last_value, reward,  obs)

                last_obs = obs
                last_action = action
                last_value = value
                obs, reward, done, info = env.step(action)

                sum_of_reward += reward
                global_step += 1

                # save model
                if global_step % 10 ** 6 == 0:
                    path = os.path.join(args.outdir,
                            '{}/model.ckpt'.format(global_step))
                    saver.save(sess, path)

                # the end of episode
                if done:
                    summary, _ = sess.run(
                        [merged, reward_summary],
                        feed_dict={reward_summary: sum_of_reward}
                    )
                    train_writer.add_summary(summary, global_step)
                    agent.stop_episode(
                            last_obs, last_action, last_value, reward)
                    print(
                        'Episode: {}, Step: {}: Reward: {}'.format(
                        episode,
                        global_step,
                        sum_of_reward
                    ))
                    episode += 1
                    break

            # backup current situation
            backup.save(i, sum_of_reward, reward,
                        obs, last_obs, last_action, last_value, done)

            # append data for training
            training_data.append(agent.get_training_data())

        # train network
        obs = []
        actions = []
        returns = []
        deltas = []
        for o, a, r, d in training_data:
            obs.extend(o)
            actions.extend(a)
            returns.extend(r)
            deltas.extend(d)
        agent.train(obs, actions, returns, deltas)

        if args.final_steps < global_step:
            break

if __name__ == '__main__':
    main()
