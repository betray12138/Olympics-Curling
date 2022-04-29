import argparse
import datetime
import math
import torch
import numpy as np
from collections import deque, namedtuple

from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
from env.chooseenv import make
from rl_trainer.log_path import *
from rl_trainer.algo.rule import my_controller
from rl_trainer.algo.random import random_agent

parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="olympics-curling", type=str)
parser.add_argument('--algo', default="ppo", type=str, help="ppo")
parser.add_argument('--controlled_player', default=0, help="0(agent purple) or 1(agent green)")
parser.add_argument('--max_episodes', default=1500, type=int)
parser.add_argument('--opponent', default="random", help="random or runX")
parser.add_argument('--opponent_load_episode', default=None)

parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--render', action='store_true')  # 加是true；不加为false

parser.add_argument("--save_interval", default=100, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
parser.add_argument("--load_run", default=1, type=int)
parser.add_argument("--load_episode", default=1500, type=int)

# 更换敌方我方颜色，更换剩余顺序

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)

def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')
    RENDER = args.render

    env = make(args.game_name)      # olympics-curling

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')

    ctrl_agent_index = int(args.controlled_player)

    print(f'Agent control by the actor: {ctrl_agent_index}')

    width = env.env_core.view_setting['width'] + 2 * env.env_core.view_setting['edge'] # 700
    height = env.env_core.view_setting['height'] + 2 * env.env_core.view_setting['edge'] # 700
    print(f'Game board width: {width}')
    print(f'Game board height: {height}')

    act_dim = env.action_dim
    obs_dim = 30 * 30
    print(f'action dimension: {act_dim}')
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)
    record_win = deque(maxlen = 100)
    record_win_op = deque(maxlen = 100)

    opponent_agent = random_agent()

    episode = 0
    train_count = 0

    while episode < args.max_episodes:
        state = env.reset()
        if RENDER:
            env.env_core.render()
        obs_ctrl_agent = np.array(state[ctrl_agent_index]['obs'])
        obs_oppo_agent = np.array(state[1-ctrl_agent_index]['obs'])

        episode += 1
        step = 0

        while True:
            oppo_action_raw, _ = opponent_agent.select_action(obs_oppo_agent.flatten(), False)
            action_opponent = oppo_action_raw
            action_ctrl = my_controller(env, obs_ctrl_agent, None)
            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]
            next_state, reward, done, _, info = env.step(action)
            next_obs_ctrl_agent = next_state[ctrl_agent_index]['obs']
            next_obs_oppo_agent = next_state[1-ctrl_agent_index]['obs']
            step += 1

            obs_oppo_agent = np.array(next_obs_oppo_agent)
            obs_ctrl_agent = np.array(next_obs_ctrl_agent)
            if RENDER:
                env.env_core.render()

            if done:
                win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                print("Episode: ", episode, "controlled agent: ", ctrl_agent_index,
                      "; win rate(controlled & opponent): ", '%.2f' % (sum(record_win)/len(record_win)),
                      '%.2f' % (sum(record_win_op)/len(record_win_op)), '; Trained episode:', train_count)
                break

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
