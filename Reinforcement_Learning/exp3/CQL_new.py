
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
TAU_ = [0.001, 0.01, 0.1]
GAMMA_ = [0.3, 0.5, 0.7]
ALPHA_ = [0.2, 0.5, 0.7]
DATA = [50,150,250,350]
# Q-network definition
class QNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super(QNetwork, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, action_dim)
 
	def forward(self, state):
		x = torch.relu(self.fc1(state))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)
# CQL algorithm
class ConservativeQLearning:
    def __init__(self, state_dim, action_dim, gamma=0.1, alpha=0.1, tau=0.1, lr=3e-4):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_q_network = QNetwork(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def update(self, batch, policy):
        states, actions, rewards, next_states, dones = batch

        # Compute target Q value
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute current Q value
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute conservative Q-learning loss
        q_diff = self.q_network(states) - policy(states)
        conservative_loss = torch.mean(
            torch.logsumexp(q_diff / self.alpha, dim=1) - torch.mean(q_diff, dim=1)
        )

        loss = nn.MSELoss()(current_q_values, target_q_values) + self.alpha * conservative_loss

        # Update Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Replay buffer to store experience
class ReplayBuffer:
	def __init__(self, capacity, state_dim):
		self.capacity = capacity
		self.buffer = []
		self.position = 0
		self.state_dim = state_dim
 
	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity
 
	def sample(self, batch_size):
		batch = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in batch])
		return (
			torch.FloatTensor(np.array(states)),
			torch.LongTensor(actions),
			torch.FloatTensor(rewards),
			torch.FloatTensor(np.array(next_states)),
			torch.FloatTensor(dones),
		)
 
	def __len__(self):
		return len(self.buffer)
 
 
# 状态归一化
class Normalizer:
	def __init__(self, state_dim):
		self.state_dim = state_dim
		self.state_mean = np.zeros(state_dim)
		self.state_std = np.ones(state_dim)
		self.count = 0
 
	def update(self, state):
		self.count += 1
		self.state_mean += (state - self.state_mean) / self.count
		self.state_std = np.sqrt(((self.count - 1) * self.state_std ** 2 + (state - self.state_mean) ** 2) / self.count)
 
	def normalize(self, state):
		return (state - self.state_mean) / (self.state_std + 1e-8)
 
import numpy as np

# Load dataset and add to ReplayBuffer
def prefill_replay_buffer(replay_buffer, dataset_path):
    # Load the .npz dataset
    data = np.load(dataset_path)
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    next_states = data['next_states']
    dones = data['dones']

    # Add each sample to the replay buffer
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        replay_buffer.push(state, action, reward, next_state, done)

    print(f"Replay buffer prefilled with {len(replay_buffer)} samples.")
# Updated training function
def train_cql_in_cartpole_with_dataset(
    env_name="CartPole-v1",
    num_episodes=500,
    batch_size=64,
    buffer_capacity=50000,
    alpha=0.1,
    gamma = 0.1,
    tau = 0.1,
    dataset_path=None
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    cql = ConservativeQLearning(state_dim, action_dim,  lr=1e-4,alpha=alpha, gamma = gamma,tau = tau )
    replay_buffer = ReplayBuffer(buffer_capacity, state_dim)

    # Pre-fill replay buffer with dataset if provided
    if dataset_path is not None:
        prefill_replay_buffer(replay_buffer, dataset_path)

    policy = lambda x: cql.q_network(x)
    total_rewards = []

    # Train using environment interaction
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = cql.q_network(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done, _, __ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                cql.update(batch, policy)
                cql.soft_update()

        total_rewards.append(episode_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}")

    return total_rewards

# CartPole environment CQL training
def train_cql_in_cartpole(env_name="CartPole-v1", num_episodes=500, batch_size=64, buffer_capacity=10000, alpha=1.0):
	env = gym.make(env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
 
	cql = ConservativeQLearning(state_dim, action_dim, alpha=alpha, lr=1e-4)
 
 
	# 初始化ReplayBuffer时增加容量
	buffer_capacity = 50000  # 增大容量至50000
	replay_buffer = ReplayBuffer(buffer_capacity, state_dim)
 
	policy = lambda x: cql.q_network(x)
	total_rewards = []
 
 
	# 使用Normalizer对状态进行归一化
	normalizer = Normalizer(state_dim)
	for episode in range(num_episodes):
		state, _ = env.reset()
		episode_reward = 0
		done = False
 
		while not done:
			state_tensor = torch.FloatTensor(state).unsqueeze(0)
			q_values = cql.q_network(state_tensor)
			action = torch.argmax(q_values).item()
 
			next_state, reward, done, _, __ = env.step(action)
			replay_buffer.push(state, action, reward, next_state, done)
			episode_reward += reward
			state = next_state
 
			if len(replay_buffer) > batch_size:
				batch = replay_buffer.sample(batch_size)
				cql.update(batch, policy)
				cql.soft_update()
 
		total_rewards.append(episode_reward)
		if episode % 10 == 0:
			print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}")
 
		# if np.mean(total_rewards[-10:]) > 195:  # Terminate early if environment solved
		# 	print("Solved CartPole!")
		# 	break
 
	return total_rewards
 


# Plot the rewards over time
import matplotlib.pyplot as plt

for TAU in TAU_:
    for GAMMA in GAMMA_:
        for ALPHA in ALPHA_:
            for dataset_path in DATA:
                
                rewards = train_cql_in_cartpole_with_dataset(
                    num_episodes=500,
                    dataset_path=f"dataset_episode_{dataset_path}.npz"
                )
                des = f"tau = {TAU}, gamma = {GAMMA}, alphs = {ALPHA}, data = {dataset_path}"
                
                plt.figure(figsize=(12, 7))  # 调整图片宽度，更宽一些
                plt.plot(
                    rewards,
                    color='blueviolet', 
                    marker='o',
                    linewidth=1,  # 线条更细
                    markersize=4   # 标记更小
)
                plt.title(f"CQL Training ({des})")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.savefig(f"{des}-CQL-training_progress.png")


