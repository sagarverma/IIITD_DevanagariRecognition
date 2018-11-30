import torch
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.schedule import LinearSchedule
from read_write import Environment

BATCH_SIZE = 2048
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 100000
LEARNING_STARTS = 500
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.001
ALPHA = 0.95
EPS = 0.01

num_actions1 = 2
num_actions2 = 27

use_gpu = torch.cuda.is_available()
DEVICE = 0

class_map = {chr(x): x-97 for x in range(97,97+26)}
class_map[' '] = 26
inv_class_map = {x-97: chr(x) for x in range(97,97+26)}
inv_class_map[26] = ' '

data_dir = '../../datasets/english-words/'
train_csv = 'train_words_alpha.txt'
test_csv = 'test_words_alpha.txt'
weights_dir = '../../weights/'

def main(env):

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        num_actions1=num_actions1,
        num_actions2=num_actions2
    )

if __name__ == '__main__':
    # Get Atari games.
    env = Environment(data_dir + train_csv, num_actions1)

    main(env)
