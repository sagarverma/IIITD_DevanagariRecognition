"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}

def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """

    ###############
    # BUILD MODEL #
    ###############

    img_h, img_w, img_c = 32, 120, 1
    input_arg = frame_history_len * img_c
    num_actions1 = 2
    num_actions2 = 27

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
            # Use volatile = True if variable is only used in inference mode, i.e. don't save the history
            out1, out2 = model(Variable(obs, volatile=True))
            return out1.data.max(1)[1].cpu(), out2.data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions1)]]), torch.IntTensor([[random.randrange(num_actions2)]])

    # Initialize target q function and q function
    Q = q_func().cuda(0).type(dtype)
    target_Q = q_func().cuda(0).type(dtype)

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    for t in count():

        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        last_idx = replay_buffer.store_frame(last_obs)
        # encode_recent_observation will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        recent_observations = replay_buffer.encode_recent_observation()

        # Choose random action if not yet start learning
        if t > learning_starts:
            action1, action2 = select_epilson_greedy_action(Q, recent_observations, t)
            action1 = action1.data.cpu().numpy()[0][0]
            action2 = action2.data.cpu().numpy()[0][0]
        else:
            action1, action2 = random.randrange(num_actions1), random.randrange(num_actions2)
        # Advance one step
        obs, reward, done = env.step(action1, action2)
        # env.render()
        # clip rewards between -1 and 1
        # reward = max(-1.0, min(reward, 1.0))
        # Store other info in replay memory
        replay_buffer.store_effect(last_idx, action1, action2, reward, done)
        # Resets the environment when reaching an episode boundary.
        if done:
            obs = env.reset()
        last_obs = obs

        ### Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act1_batch, act2_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            # Convert numpy nd_array to torch variables for calculation
            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
            act1_batch = Variable(torch.from_numpy(act1_batch).long())
            act2_batch = Variable(torch.from_numpy(act2_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype))
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

            if USE_CUDA:
                act1_batch = act1_batch.cuda()
                act2_batch = act2_batch.cuda()
                rew_batch = rew_batch.cuda()

            # Compute current Q value, q_func takes only stateif stopping_criterion is not None and stopping_criterion(env):
            # break and output value for every state-action pair
            # We choose Q based on action taken.
            q1, q2 =   Q(obs_batch)
            current_Q1_values = q1.gather(1, act1_batch.unsqueeze(1))
            current_Q2_values = q2.gather(1, act2_batch.unsqueeze(1))
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            tq1, tq2 = target_Q(next_obs_batch)
            next_max_q1 = tq1.detach().max(1)[0]
            next_max_q2 = tq2.detach().max(1)[0]
            next_Q1_values = not_done_mask * next_max_q1
            next_Q2_values = not_done_mask * next_max_q2
            # Compute the target of the current Q values
            target_Q1_values = rew_batch + (gamma * next_Q1_values)
            target_Q2_values = rew_batch + (gamma * next_Q2_values)
            # Compute Bellman error
            bellman_error1 = target_Q1_values - current_Q1_values
            bellman_error2 = target_Q2_values - current_Q2_values
            bellman_error = bellman_error1 + bellman_error2
            # clip the bellman error between [-1 , 1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # run backward pass
            current_Q_values = current_Q1_values + current_Q2_values
            print current_Q_values.size(), d_error.size()
            current_Q_values.backward(d_error.data.unsqueeze(1))

            # Perfom the update
            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

        ### 4. Log progress and keep track of statistics
        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        # if len(episode_rewards) > 0:
        #     mean_episode_reward = np.mean(episode_rewards[-100:])
        # if len(episode_rewards) > 100:
        #     best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        #
        # Statistic["mean_episode_rewards"].append(mean_episode_reward)
        # Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)
        #
        # if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
        #     print("Timestep %d" % (t,))
        #     print("mean reward (100 episodes) %f" % mean_episode_reward)
        #     print("best mean reward %f" % best_mean_episode_reward)
        #     print("episodes %d" % len(episode_rewards))
        #     print("exploration %f" % exploration.value(t))
        #     sys.stdout.flush()
        #
        #     # Dump statistics to pickle
        #     with open('statistics.pkl', 'wb') as f:
        #         pickle.dump(Statistic, f)
        #         print("Saved to %s" % 'statistics.pkl')
