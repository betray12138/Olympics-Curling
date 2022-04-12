import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from rgbd_resnet18 import *
import torchvision.models.vgg as vgg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 512)
		self.l2 = nn.Linear(512, 256)
		self.l3 = nn.Linear(256, 128)
		self.l4 = nn.Linear(128, action_dim)
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		return torch.tanh(self.l4(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 512)
		self.l2 = nn.Linear(512, 256)
		self.l3 = nn.Linear(256, 128)
		self.l4 = nn.Linear(128, 1)

		# Q2 architecture
		self.l5 = nn.Linear(state_dim + action_dim, 512)
		self.l6 = nn.Linear(512, 256)
		self.l7 = nn.Linear(256, 128)
		self.l8 = nn.Linear(128, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = F.relu(self.l3(q1))
		q1 = self.l4(q1)

		q2 = F.relu(self.l5(sa))
		q2 = F.relu(self.l6(q2))
		q2 = F.relu(self.l7(q2))
		q2 = self.l8(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = F.relu(self.l3(q1))
		q1 = self.l4(q1)
		return q1

# class PreProcess(nn.Module):
# 	def __init__(self) -> None:
# 		super().__init__()
# 		self.rgb = vgg.VGG(vgg.make_layers(vgg.cfgs['D'], batch_norm=True))
# 		self.rgb.load_state_dict(torch.load("vgg16_method2.pth"))
# 		self.rgb.classifier = nn.Sequential()

# 		self.rgb_link = nn.Linear(25088, 512)

# 		self.output_size = 512 + 20 * 8
	
# 	def forward(self, state):
# 		partone = self.rgb(state[:, 0: 128 * 128 * 3].view(-1, 3, 128, 128))
# 		partone = self.rgb_link(partone)
# 		linkvec = torch.cat((partone, state[:, 128 * 128 * 3: ]), dim = 1)

# 		#linkvec = torch.cat((torch.zeros([state.shape[0], 512]).to(device), state), dim = 1)

# 		return linkvec


class TD3(object):
	def __init__(self, 
	img_size,
    state_dim, 
    action_dim, 
	writter, 
    discount=0.99, 
    tau=0.005, 
    policy_noise=0.05, 
    noise_clip=0.1, 
    policy_freq=5):
		self.img_size = img_size

		self.actor = Actor(state_dim, action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam([
			{'params': self.actor.parameters(), 'lr': 3e-4}])

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam([
			{'params': self.critic.parameters(), 'lr': 3e-4}])

		self.writer = writter

		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		# rgbd_feature = self.rgbd(state[:, 0:self.img_size * self.img_size * 3].view(-1, 3, self.img_size, self.img_size), 
		# 		state[:, self.img_size * self.img_size * 3:self.img_size * self.img_size * 4].view(-1, self.img_size, self.img_size))[0]
		# rgbd_feature = rgbd_feature.reshape(rgbd_feature.shape[0], -1)
		# current_state = torch.cat((rgbd_feature, state[:, self.img_size * self.img_size * 4:]), dim=1)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# with torch.no_grad():
		# 	cur_rgbd_feature = self.rgbd(state[:, 0:self.img_size * self.img_size * 3].view(-1, 3, self.img_size, self.img_size), 
		# 		state[:, self.img_size * self.img_size * 3:self.img_size * self.img_size * 4].view(-1, self.img_size, self.img_size))[0]
		# 	cur_rgbd_feature = cur_rgbd_feature.reshape(cur_rgbd_feature.shape[0], -1)
		# 	current_state = torch.cat((cur_rgbd_feature, state[:, self.img_size * self.img_size * 4:]), dim=1)

		# 	next_rgbd_feature = self.rgbd(next_state[:, 0:self.img_size * self.img_size * 3].view(-1, 3, self.img_size, self.img_size), 
		# 		next_state[:, self.img_size * self.img_size * 3:self.img_size * self.img_size * 4].view(-1, self.img_size, self.img_size))[0]
		# 	next_rgbd_feature = next_rgbd_feature.reshape(next_rgbd_feature.shape[0], -1)
		# 	current_next_state = torch.cat((next_rgbd_feature, next_state[:, self.img_size * self.img_size * 4:]), dim=1)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-1, 1)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward(retain_graph=True)
		self.critic_optimizer.step()
		self.writer.add_scalar("critic_loss", critic_loss.item(), self.total_it)

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			self.writer.add_scalar("actor_loss", actor_loss.item(), self.total_it)

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		#torch.save(self.rgbd.state_dict(), "carla/" + filename + "_rgbd")
		#torch.save(self.preprocess.state_dict(), "carla/" + filename + "_preprocess")

		torch.save(self.critic.state_dict(), "carla/" + filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), "carla/" + filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), "carla/" + filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), "carla/" + filename + "_actor_optimizer")


	def load(self, filename):
		#self.rgbd.load_state_dict(torch.load("carla/" + filename + "_rgbd"))
		#self.preprocess.load_state_dict(torch.load("carla/" + filename + "_preprocess"))

		self.critic.load_state_dict(torch.load("carla/" + filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load("carla/" + filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load("carla/" + filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load("carla/" + filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)