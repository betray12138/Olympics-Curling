import os
from os import path
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size = 64):
        super(Actor, self).__init__()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, state):
        x = F.relu(self.linear_in(state))
        action_prob = F.softmax(self.action_head(x), dim = 1)
        return action_prob

class Critic(nn.Module):
	def __init__(self, state_space, action_space, hidden_size = 64):
		super(Critic, self).__init__()

		# Q1 architecture
        self.linear_in_1 = nn.Linear(state_space + action_space, hidden_size)
        self.state_value_1 = nn.Linear(hidden_size, 1)

		# Q2 architecture
        self.linear_in_2 = nn.Linear(state_space + action_space, hidden_size)
        self.state_value_2 = nn.Linear(hidden_size, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.linear_in_1(sa))
		q1 = self.state_value_1(q1)

		q2 = F.relu(self.linear_in_2(sa))
		q2 = self.state_value_2(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.linear_in_1(sa))
		q1 = self.state_value_1(q1)
        return q1

class Args:
    discount = 0.99
	tau = 0.005
	policy_noise = 0.05
	noise_clip = 0.1
	policy_freq = 5

    td3_update_time = 10
    buffer_capacity = 1000
    batch_size = 32
    lr = 0.0001

    action_space = 36
    state_space = 900
    train_count = 0


args = Args()
device = 'cpu'

class TD3:
    discount = args.discount
    tau = args.tau
    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    policy_freq = args.policy_freq

    td3_update_time = args.td3_update_time
    buffer_capacity = args.buffer_capacity
    batch_size = args.batch_size
    lr = args.lr

    action_space = args.action_space
    state_space = args.state_space
    
    use_cnn = False

    def __init__(self, run_dir=None):
        super(PPO, self).__init__()
        self.args = args
        self.actor = Actor(self.state_space, self.action_space).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = self.lr)

        self.critic = Critic(self.state_space, self.action_space).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.lr)

        self.buffer = []
        self.counter = 0
        self.training_step = 0

        if run_dir is not None:
            self.writer = SummaryWriter(os.path.join(run_dir, "TD3 training loss at {}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))
        self.IO = True if (run_dir is not None) else False

    def select_action(self, state, train = True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state).to(device)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
        return action.item(), action_prob[ : , action.item()].item()

    def get_value(self, state, action):
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        with torch.no_grad():
            value = self.critic(state, action)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, episode):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype = torch.float).to(device)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype = torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index].squeeze(1))
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index].squeeze(1)).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

                if self.IO:
                    self.writer.add_scalar('loss/policy loss', action_loss.item(), self.training_step)
                    self.writer.add_scalar('loss/critic loss', value_loss.item(), self.training_step)

        self.clear_buffer()

    def clear_buffer(self):
        del self.buffer[ : ]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)

    def load(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        print("run_dir: ", run_dir)
        base_path = os.path.dirname(os.path.dirname(__file__))
        print("base_path: ", base_path)
        algo_path = os.path.join(base_path, 'models/olympics-curling')
        run_path = os.path.join(algo_path, run_dir)
        run_path = os.path.join(run_path, 'trained_model')
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location = device)
            critic = torch.load(model_critic_path, map_location = device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

