import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.rl.replay_buffer import ReplayBuffer
from network import make_actor_network, make_critic_network
from agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--final-steps', type=int, default=10 ** 7)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
    if args.logdir is None:
        args.logdir = os.path.join(os.path.dirname(__file__), 'logs')

    env = gym.make(args.env)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = make_actor_network([30])
    critic = make_critic_network()
    replay_buffer = ReplayBuffer(10 ** 5)

    sess = tf.Session()
    sess.__enter__()

    agent = Agent(actor, critic, obs_dim, n_actions, replay_buffer)

    initialize()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    reward_summary = tf.placeholder(tf.int32, (), name='reward_summary')
    tf.summary.scalar('reward_summary', reward_summary)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.logdir, sess.graph)

    global_step = 0
    episode = 0

    while True:
        reward = 0
        done = False
        sum_of_rewards = 0
        step = 0
        state = env.reset()

        while True:
            if args.render:
                env.render()

            if done:
                summary, _ = sess.run([merged, reward_summary], feed_dict={reward_summary: sum_of_rewards})
                train_writer.add_summary(summary, global_step)
                agent.stop_episode_and_train(state, reward, done=done)
                break

            action = agent.act_and_train(state, reward, episode)

            state, reward, done, info = env.step(action)

            sum_of_rewards += reward
            step += 1
            global_step += 1

            if global_step % 10 ** 6 == 0:
                path = os.path.join(args.outdir, '{}/model.ckpt'.format(global_step))
                saver.save(sess, path)

        episode += 1

        print('Episode: {}, Step: {}: Reward: {}'.format(
                episode, global_step, sum_of_rewards))

        if args.final_steps < global_step:
            break

if __name__ == '__main__':
    main()
