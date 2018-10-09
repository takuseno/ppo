## PPO
Proximal Policy Optimization implementation with Tensorflow.

https://arxiv.org/pdf/1707.06347.pdf

This repository has been much updated from commit id `a4fbd383f0f89ce2d881a8b78d6b8a03294e5c7c` .
New PPO requires a new dependency, [rlsaber](https://github.com/imai-laboratory/rlsaber) which is my utility repository that can be shared across different algorithms.

Some of my design follow [OpenAI baselines](https://github.com/openai/baselines).
But, I used as many default tensorflow packages as possible unlike baselines, that makes my codes easier to be read.

In addition, my PPO automatically switches between continuous action-space and discrete action-space depending on environments.
If you want to change hyper parameters, check `atari_constants.py` or `box_constants.py`, which will be loaded depending on environments too.

## requirements
- Python3

## dependencies
- tensorflow
- gym[atari]
- opencv-python
- git+https://github.com/imai-laboratory/rlsaber

## usage
### training
```
$ python train.py [--env env-id] [--render] [--logdir log-name]
```
example
```
$ python train.py --env BreakoutNoFrameskip-v4 --logdir breakout
```

### playing
```
$ python train.py --demo --load results/path-to-model [--env env-id] [--render]
```
example
```
$ python train.py --demo --load results/breakout/model.ckpt-xxxx --env BreakoutNoFrameskip-v4 --render
```

### performance examples
#### Pendulumn-v0
![image](https://user-images.githubusercontent.com/5235131/46388030-e4f72980-c704-11e8-9d76-1790dcb88067.png)

#### BreakoutNoFrameskip-v4
![image](https://user-images.githubusercontent.com/5235131/46402330-6321f300-c73a-11e8-9b46-46959bce4c3d.png)


### implementation
This is inspired by following projects.

- [DQN](https://github.com/imai-laboratory/dqn)
- [OpenAI Baselines](https://github.com/openai/baselines)

## License
This repository is MIT-licensed.
