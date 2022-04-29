import math
from queue import Queue
from tkinter import E
import numpy as np
import sys
import os

log = False
sys.stdout = sys.__stdout__ if log else open(os.devnull, 'w')

class rule_agent:
    def __init__(self):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.obs_dim = 30
        self.ts = 0
        self.gamma = 0.98
        self.tau = 0.1
        self.v = 0.
        self.x = 300.
        self.y = 150.
        self.base_point = [300, 150]
        self.oppo_point = [0, 0]
        self.obs = np.zeros((30, 30), dtype = np.float64)
        self.angles = []
        self.left_or_right = 0 # 0左1右
        self.begin_force = 100
        self.final_force = 200
        self.term = 10
        # 阶段 前进、后退、转向、向右、转向、前进
        self.stages = [self.term, 2 * self.term + 3, self.term + 1, 1, 3, self.term, self.term, 1, 4]
        self.f_200 = 56

    def reset(self):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.obs_dim = 30
        self.ts = 0
        self.gamma = 0.98
        self.tau = 0.1
        self.v = 0.
        self.x = 300.
        self.y = 150.
        self.base_point = [300, 150]
        self.oppo_point = [0, 0]
        self.obs = np.zeros((30, 30), dtype = np.float64)
        self.angles = []
        self.left_or_right = 0 # 0左1右
        self.begin_force = 100
        self.final_force = 200
        self.term = 10
        # 阶段 前进、后退、转向、向右、转向、前进
        self.stages = [self.term, 2 * self.term + 3, self.term + 1, 1, 3, self.term, self.term, 1, 4]
        self.f_200 = 56

    def cal_oppo_point(self, idxes):
        min_dis, min_i = 1e6, -1
        for i, idx in enumerate(idxes):
            self.oppo_point[1] = self.base_point[1] + (29 - idx[0]) * 10 + 5
            if idx[1] <= 14:
                self.oppo_point[0] = self.base_point[0] + (14 - idx[1]) * 10 + 5
            else:
                self.oppo_point[0] = self.base_point[0] - (idx[1] - 15) * 10 - 5
            if math.sqrt((300 - agent.oppo_point[0]) ** 2 + (500 - agent.y) ** 2) <= min_dis:
                min_dis = math.sqrt((300 - agent.oppo_point[0]) ** 2 + (500 - agent.y) ** 2)
                min_i = i
        self.oppo_point[1] = self.base_point[1] + (29 - idxes[min_i][0]) * 10 + 5
        if idxes[min_i][1] <= 14:
            self.oppo_point[0] = self.base_point[0] + (14 - idxes[min_i][1]) * 10 + 5
        else:
            self.oppo_point[0] = self.base_point[0] - (idxes[min_i][1] - 15) * 10 - 5
        print('敌方的位置', self.oppo_point[0], self.oppo_point[1])

    def get_terms(self, stages):
        terms = [0]
        for stage in stages:
            terms.append(terms[-1] + stage)
        return terms

agent = rule_agent()

def my_controller(env, observation, action_space, is_act_continuous = False):
    obs = np.squeeze(observation)
    agent.last_obs = obs
    print(env.env_core.agent_pos)
    terms = agent.get_terms(agent.stages)
    if terms[0] < agent.ts <= terms[1]:
        agent_action = [[agent.begin_force], [0]]

    elif terms[1] < agent.ts <= terms[2]:
        if agent.ts == int((terms[1] + terms[2]) / 2): # 达到最近点，记录位置信息
            agent.base_point = [300, agent.y]
            agent.obs = obs
            idxes = np.argwhere(agent.obs == 1)
            print(obs)
            if len(idxes):
                agent.cal_oppo_point(idxes)
            else:
                agent.oppo_point = [-1, -1]
            agent.left_or_right = 0 if agent.oppo_point[0] <= 300 else 1
            # input()
        agent_action = [[- agent.begin_force], [0]]

    elif terms[2] < agent.ts <= terms[3]:
        agent_action = [[agent.begin_force], [0]]

    elif terms[3] < agent.ts <= terms[4]: # 速度降为0
        print('***', agent.v)
        agent_action = [[- agent.v * agent.gamma * 10], [0]]

    elif terms[4] < agent.ts <= terms[5]:
        if agent.left_or_right:
            agent_action = [[0], [-30]]
        else:
            agent_action = [[0], [30]]

    elif terms[5] < agent.ts <= terms[6]:
        agent_action = [[agent.begin_force], [0]]

    elif terms[6] < agent.ts <= terms[7]:
        agent_action = [[- agent.begin_force], [0]]

    elif terms[7] < agent.ts <= terms[8]: # 速度降为0
        print('***', agent.v)
        oppo_point = agent.oppo_point if agent.oppo_point[0] != -1 else [300, 500]
        agent_action = [[- agent.v * agent.gamma * 10], [0]]

        dis = math.sqrt((agent.x - oppo_point[0]) ** 2 + (oppo_point[1] - agent.y) ** 2)
        angle = math.acos((agent.x - oppo_point[0]) / dis) * 180 / math.pi
        print(angle)
        if agent.left_or_right:
            angle = 180 - angle
            while angle >= 30:
                agent.angles.append(30)
                angle -= 30
            if angle > 0:
                agent.angles.append(angle)
        else:
            angle = - angle
            while angle <= -30:
                agent.angles.append(-30)
                angle += 30
            if angle < 0:
                agent.angles.append(angle)

        agent.stages[8] = len(agent.angles)
        terms = agent.get_terms(agent.stages)

    elif terms[8] < agent.ts <= terms[9]:
        agent_action = [[0], [agent.angles[-1]]]
        agent.angles.pop()

    else:
        if agent.oppo_point[0] == -1 and agent.oppo_point[1] == -1:
            agent_action = [[agent.f_200], [0]]
        else:
            agent_action = [[agent.final_force], [0]]

    if terms[0] <= agent.ts <= terms[4]: # 前后走
        agent.y = agent.y + agent.v * agent.tau
        agent.v = agent.v * agent.gamma + agent_action[0][0] * agent.tau

    elif terms[4] < agent.ts <= terms[8]: # 左右走
        if agent.left_or_right:
            agent.x = agent.x + agent.v * agent.tau
        else:
            agent.x = agent.x - agent.v * agent.tau
        agent.v = agent.v * agent.gamma + agent_action[0][0] * agent.tau

    agent.ts += 1
    if (obs == np.zeros((agent.obs_dim, agent.obs_dim), dtype = np.float64) - 1.).all():
        agent.reset()

    print(agent.ts)
    print(agent.x, agent.y)
    return agent_action

# todo
# 判断对方和敌方
# 找到球的中心
# 能量设置，代替对方位置
# 直线防止把自己打中
# 为什么有时候会直接出去？
# 计算多少距离，然后进行平行打击
