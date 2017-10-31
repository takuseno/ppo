import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from network import make_actor_network, make_critic_network
from agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = make_actor_network([30])
    critic = make_critic_network()

    sess = tf.Session()
    sess.__enter__()

    agent = Agent(actor, critic, obs_dim, n_actions, None)

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    global_step = 0
    episode = 0

    while True:
        sum_of_rewards = 0
        done = False
        step = 0
        state = env.reset()

        while True:
            if args.render:
                env.render()

            action = agent.act(state)

            if done:
                break

            state, reward, done, info = env.step(action)

            sum_of_rewards += reward
            step += 1
            global_step += 1

        episode += 1

        print('Episode: {}, Step: {}: Reward: {}'.format(
                episode, global_step, sum_of_rewards))

if __name__ == '__main__':
    main()
