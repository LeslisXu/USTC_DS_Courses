import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import collections

# hyper-parameters
EPISODES = 2000                 # 训练/测试幕数
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
SAVING_IETRATION = 1000         # 保存Checkpoint的间隔
MEMORY_CAPACITY = 10000         # Memory的容量
MIN_CAPACITY = 500              # 开始学习的下限
Q_NETWORK_ITERATION = 10        # 同步target network的间隔
EPSILON = 0.01                  # epsilon-greedy
SEED = 0
TEST = True


dqn_net_type_list = ["DQN","DuelingDQN"]
dqn_type_list = ['DQN',"DDQN"]
# python DQN_new.py
choice = 4
if choice == 1:
    dqn_type,dqn_net_type = dqn_type_list[1],dqn_net_type_list[0]
elif choice == 2:
    dqn_type,dqn_net_type = dqn_type_list[1],dqn_net_type_list[1]
elif choice == 3:    
    dqn_type,dqn_net_type = dqn_type_list[0],dqn_net_type_list[0]
elif choice == 4:
    dqn_type,dqn_net_type = dqn_type_list[0],dqn_net_type_list[1]

env_id = 3

DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu' # cuda:0
if env_id==1:
    env = gym.make('CartPole-v1', render_mode="human" if TEST else None)
if env_id==2:
    env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
if env_id==3:
    env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)

MODEL_PATH = '3_DQN_DuelingDQN/566000.pth'
SAVE_PATH_PREFIX = 'save_results/'
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

NUM_ACTIONS = env.action_space.n  # 2
NUM_STATES = env.observation_space.shape[0]  # 4
ENV_A_SHAPE = 0 if np.issubdtype(type(env.action_space.sample()), np.integer) else env.action_space.sample().shape  # 0, to confirm the shape

class Model(nn.Module):
    def __init__(self, num_inputs=4):
        super(Model, self).__init__()
        self.linear = nn.Linear(NUM_STATES, 512)
        self.linear2 = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
class DuelingModel(nn.Module):
    def __init__(self, num_inputs=4):
        super(DuelingModel, self).__init__()
        self.linear = nn.Linear(NUM_STATES, 512)
        self.linear_action = nn.Linear(512, NUM_ACTIONS)
        self.linear_value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        action = self.linear_action(x)
        value = self.linear_value(x)
        Q = value + (action - action.mean(dim=-1, keepdim=True))
        return Q

class Data:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done





class Memory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        # TODO
        self.buffer.append(data)

    def get(self, batch_size):
        # TODO
        samples = random.sample(self.buffer, batch_size)

        return samples


class DQN():
    """docstring for DQN"""
    def __init__(self,dqn_type='DQN',dqn_net_type='DQN',m=1):
        super(DQN, self).__init__()
        self.dqn_type = dqn_type
        self.dqn_net_type = dqn_net_type
        if dqn_net_type=='DQN':
            self.eval_net, self.target_net = Model().to(device), Model().to(device)
        if dqn_net_type == 'DuelingDQN':
            self.eval_net, self.target_net = DuelingModel().to(device), DuelingModel().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.dqn_type = dqn_type
        self.m = m  # m步 dqn
        self.factior = GAMMA
    def choose_action(self, state, EPSILON = 1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        else: 
            # random policy
            action = np.random.randint(0,NUM_ACTIONS)  # int random number
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # TODO
        # sample batch from memory
        batch_memory = self.memory.get(BATCH_SIZE)
        batch_state = torch.tensor(np.array([sample.state for sample in batch_memory]), dtype=torch.float).to(device)
        batch_action = torch.tensor(np.array([sample.action for sample in batch_memory]), dtype=torch.long).view(-1, 1).to(device)
        batch_reward = torch.tensor(np.array([sample.reward for sample in batch_memory]), dtype=torch.float).view(-1, 1).to(device)
        batch_next_state = torch.tensor(np.array([sample.next_state for sample in batch_memory]), dtype=torch.float).to(device)
        batch_done = torch.tensor(np.array([sample.done for sample in batch_memory]), dtype=torch.float).view(-1, 1).to(device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # qt(st, at) 并取相应动作的q



        if self.dqn_type == 'DDQN':  # double dqn
            max_next_action = self.eval_net(batch_next_state).max(1)[1].view(-1, 1)  # 选最大价值的动作
            max_next_q_values = self.factior * self.target_net(batch_next_state).gather(1, max_next_action)  # 乘以折扣因子
        if self.dqn_type == 'DQN':
            q_next = self.target_net(batch_next_state).detach()  # qt+1(st+1, at+1)
            max_next_q_values =  self.factior * q_next.max(1)[0].view(-1,1)

        q_target = batch_reward + (1 - batch_done)* max_next_q_values
        # print(())
        # print(q_target)
        # print(q_eval)
        # quit()
        loss = self.loss_func(q_eval, q_target)
        # print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file, map_location='cuda:0'))
        self.target_net.load_state_dict(torch.load(file, map_location='cuda:0'))

def main():

    dqn = DQN(dqn_type=dqn_type,dqn_net_type=dqn_net_type)  # DQN
    # dqn = DQN(dqn_type="DDQN", dqn_net_type="DuelingDQN")    # DuelingDDQN
    # dqn = DQN(dqn_type="DQN", dqn_net_type="DuelingDQN")    # DuelingDQN
    # dqn = DQN(dqn_type="DDQN", dqn_net_type="DQN")           # DDQN

    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}')
    EPISODES = 2000
    if TEST:
        dqn.load_net(MODEL_PATH)
        EPISODES=1
    for i in range(EPISODES):
        print("EPISODE: ", i)
        state, info = env.reset(seed=SEED)

        ep_reward = 0
        while True:
            action = dqn.choose_action(state=state, EPSILON=EPSILON if not TEST else 0)  # choose best action
            next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward
            dqn.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            if TEST:
                env.render()
            if dqn.memory_counter >= MIN_CAPACITY and not TEST:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                if TEST:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                break
            state = next_state
        writer.add_scalar('reward', ep_reward, global_step=i)


if __name__ == '__main__':
    main()