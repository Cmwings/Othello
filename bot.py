import torch
import torch.nn as nn
import numpy as np
import os
import sys
import random

# Define the neural network architecture used as the brain for the DQN
class Brain(nn.Module):
    def __init__(self, a_dim=64, s_dim=64):
        super(Brain, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, a_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x

# Define the Deep Q-Network (DQN) class
class DQN(nn.Module):
    def __init__(self, player="Black", device=torch.device("cuda"), model_name="Black_Model"):
        super(DQN, self).__init__()
        self.model_name = model_name
        assert player in ("Black", "White"), "Wrong player color"
        self.id = player
        self.double_q = True
        self.prioritized = True

        self.a_dim = 64
        self.s_dim = 64
        self.device = device
        self.gamma = 0.95  
        self.alpha1 = 0.01
        self.alpha2 = 0.01 
        self.epsilon_max = 1.0
        self.batch_size = 64
        self.epsilon_increment = 0.0005
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
       
        self.learn_step_counter = 0  
        self.replace_target_iter = 200

        # Initialize the evaluation and target networks
        self.brain_evl = Brain(a_dim=self.a_dim, s_dim=self.s_dim).to(device)
        if self.id == "White": 
            self.brain_tgt = Brain(a_dim=self.a_dim, s_dim=self.s_dim).to(device)
            self.memory_size = 10000
            self.memory_counter = 0
           
            if self.prioritized:  
                self.memory = Memory(self.memory_size)
            else:
                self.memory = np.zeros((self.memory_size, self.s_dim + 1 + 1 + 1 + self.s_dim + self.a_dim), dtype=np.float)
            self.opt = torch.optim.Adam(self.brain_evl.parameters(), lr=1e-3, weight_decay=0.1)
            self.critic = nn.MSELoss()

    # Load a pre-trained model
    def load_model(self, name="Best_Model"):
        if not os.path.exists(os.path.join("models", name)):
            sys.exit("cannot load %s" % name)
        self.brain_evl.load_state_dict(torch.load(os.path.join("models", name)))

    # Choose an action based on the current observation and possible actions
    def choose_action(self, obs, a_possible):
        self.brain_evl.eval()
        with torch.no_grad():
            obs = torch.tensor(np.array(obs).ravel(), dtype=torch.float32).unsqueeze(0).to(self.device)  
            mask = torch.tensor([[True] * 64], dtype=torch.bool).to(self.device)  
            for r, c in a_possible:
                mask[0][r * 8 + c] = False 

            probs = self.brain_evl(obs)  
            probs = probs.masked_fill(mask, -1e9)
            probs = torch.softmax(probs, dim=1)  

            if np.random.uniform() < self.epsilon:
                action = torch.argmax(probs, dim=1).item()
                action = (action // 8, action % 8)
            else:
                action = random.choice(list(a_possible))
        return action

    # Store a transition in memory
    def store_transition(self, obs, a, r, done, obs_, a_possible):
        if self.id == "White":
            a_mask = np.ones(self.a_dim)
            for row, col in a_possible:
                a_mask[row * 8 + col] = 0 
            transition = np.hstack((np.array(obs).ravel(), a[0]*8+a[1], r, done, np.array(obs_).ravel(), a_mask)) 

            if self.prioritized: 
                self.memory.store(transition)
            else:
                index = self.memory_counter % self.memory_size
                self.memory[index] = transition
                self.memory_counter += 1
                if self.memory_counter == self.memory_size*3:  
                    self.memory_counter -= self.memory_size

    # Update the weights of the evaluation network
    def weights_assign(self, another: Brain):
        if self.id == "Black":
            with torch.no_grad():
                for tgt_param, src_param in zip(self.brain_evl.parameters(), another.parameters()):
                    tgt_param.data.copy_(self.alpha1 * src_param.data + (1.0 - self.alpha1) * tgt_param.data)

    # Synchronize the evaluation and target networks
    def __tgt_evl_sync(self):
        if self.id == "White":
            for tgt_param, src_param in zip(self.brain_tgt.parameters(), self.brain_evl.parameters()):
                tgt_param.data.copy_(self.alpha2 * src_param.data + (1.0 - self.alpha2) * tgt_param.data)

    # Update the evaluation network
    def learn(self):
        if self.id == "White": 
            self.brain_evl.train()
            self.brain_tgt.eval() 
            self.opt.zero_grad()
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.__tgt_evl_sync()

            if self.prioritized:
                tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
                ISWeights = torch.tensor(ISWeights, dtype=torch.float).squeeze().to(self.device)
            else:
                if self.memory_counter > self.memory_size:
                    sample_index = np.random.choice(self.memory_size, size=self.batch_size)
                else:
                    sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
                batch_memory = self.memory[sample_index]

            obs = torch.tensor(batch_memory[:, :self.s_dim], dtype=torch.float).to(self.device)
            a = torch.tensor(batch_memory[:, self.s_dim], dtype=torch.long).to(self.device)
            r = torch.tensor(batch_memory[:, self.s_dim+1], dtype=torch.float).to(self.device)
            done = torch.tensor(batch_memory[:, self.s_dim+2], dtype=torch.bool).to(self.device)
            obs_ = torch.tensor(batch_memory[:, -self.s_dim-self.a_dim: -self.a_dim], dtype=torch.float).to(self.device)

            q_eval = self.brain_evl(obs)  
            q_eval_wrt_a = torch.gather(q_eval, dim=1, index=a.view(-1, 1)).squeeze()  
            with torch.no_grad(): 
                q_next = self.brain_tgt(obs_)  
                if self.double_q:  
                    q_eval4next = self.brain_evl(obs_)  
                    max_act4next = torch.argmax(q_eval4next, dim=1)  
                    slected_q_next = torch.gather(q_next, dim=1, index=max_act4next.view(-1, 1)).squeeze() 
                    q_target = r + self.gamma * slected_q_next 
                else: 
                    q_target = r + self.gamma * torch.max(q_next, dim=1)[0]  
                q_target[done] = r[done]  

            if self.prioritized:
                with torch.no_grad():
                    abs_errors = torch.abs(q_target - q_eval_wrt_a).cpu().data.numpy()
                loss = torch.mean(ISWeights * torch.square(q_target - q_eval_wrt_a))
                self.memory.batch_update(tree_idx, abs_errors)
            else:
                loss = self.critic(q_target, q_eval_wrt_a)
                
            loss.backward()
            self.opt.step()
            self.learn_step_counter += 1
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            return loss.item()

    # Update the reward transition
    def reward_transition_update(self, reward:float):
        if self.id == "White":
            if self.prioritized:
                index = (self.memory.tree.data_pointer - 1) % self.memory_size
                self.memory.tree.data[index][self.s_dim+1] = reward
            else:
                index = (self.memory_counter - 1) % self.memory_size
                self.memory[index, self.s_dim+1] = reward

    # Save the model
    def save_model(self, name:str):
        torch.save(self.brain_evl.state_dict(), os.path.join("models", name))

    # Load a pre-trained model
    def load_model(self, name="Best_Model"):
        if not os.path.exists(os.path.join("models", name)):
            sys.exit("cannot load %s" % name)
        self.brain_evl.load_state_dict(torch.load(os.path.join("models", name)))

# Define the SumTree data structure for prioritized experience replay
class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity 
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data 
        self.update(tree_idx, p) 
        self.data_pointer += 1

        if self.data_pointer >= self.capacity: 
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
       
        while tree_idx != 0:  
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:  
            cl_idx = 2 * parent_idx + 1  
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  
                leaf_idx = parent_idx
                break
            else:  
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  

# Define the memory buffer for storing and sampling transitions for training the DQN
class Memory(object):  
    epsilon = 0.01 
    alpha = 0.6  
    beta = 0.4  
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1. 

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n 
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon 
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
