from DDQN_network import DDQN_network  # Assuming DDQN_network is in a separate file
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque

data_file = "training_data.txt"
class Agent:
    def __init__(self, env, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
        self.env = env  # Our environment
        self.epsilon = initial_epsilon # Epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = 0.96  # Discount rate
        self.lr = 0.0001  # Learning rate
        self.batch_size = 256  # Batch size
        self.episodes = 8000  # Number of episodes
        self.mem_size = 50000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device (GPU if available, otherwise CPU)
        self.Q1 = DDQN_network(frames=4, actions=env.action_space.n).to(self.device)  # Q-online
        self.Q2 = DDQN_network(frames=4, actions=env.action_space.n).to(self.device)  # Q-target
        self.optimizer = torch.optim.Adam(self.Q1.parameters(), self.lr)
        self.rewards = []  # List of rewards
        self.episode_times = [] # List of time per episode

    # Epsilon greedy algorithm
    def epsilon_greedy(self, state, Q1):
        if self.epsilon > np.random.rand():
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                action = np.argmax(Q1(state.to(Q1.device)).cpu().detach().numpy())
        return action

    def fit(self, memory):
        # Replay memory
        transitions = random.sample(memory, self.batch_size)

        state, reward, action, next_state, done = (np.array(x) for x in zip(*transitions))
        state, next_states = np.squeeze(state), np.squeeze(next_state)
        reward, done = torch.FloatTensor(reward).unsqueeze(-1).to(self.device), torch.FloatTensor(done).unsqueeze(-1).to(self.device)

        # Calculation of target values
        with torch.no_grad():
            next_actions_max = self.Q2(next_states.to(self.device)).max(1)[1].unsqueeze(-1)
            target = reward + self.gamma * self.Q2(next_states.to(self.device)).gather(1, next_actions_max) * done

        actions = torch.tensor(action).unsqueeze(-1).to(self.device)
        predictions = torch.gather(self.Q1(state.to(self.device)), dim=1, index=actions.view(-1, 1).long())

        # Calculating losses and updating gradients
        loss = F.smooth_l1_loss(predictions, target).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # Function to save best model to replay afterwards and optionally save as GIF
    def save_model(self, filename='best_model.pth'):
        torch.save(self.Q1.state_dict(), filename)
        print(f"Model saved as {filename}")

    # Function to load saved best model
    def load_model(self, filename='best_model.pth'):
        if os.path.exists(filename):
            self.Q1.load_state_dict(torch.load(filename))
            print(f"Model loaded from {filename}")
        else:
            print(f"No model found at {filename}")

    # Function to train model
    def train_model(self):
        memory = deque(maxlen=self.mem_size)
        reward_ = 0.0
        loss = 0.0

        for episode in range(self.episodes):

            # State preprocessing
            state = self.env.reset()
            state = np.array(state) if not isinstance(state, np.ndarray) else state
            assert len(state.shape) == 3
            state = np.expand_dims(np.transpose(state, (2, 0, 1)), 0)
            done = False

            # Start timer to fix time per episode
            start_time = time.time()

            while not done:
                # Epsilon-greedy algorithm
                action = self.epsilon_greedy(state, self.Q1)

                # Next state preprocessing
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.array(next_state) if not isinstance(next_state, np.ndarray) else next_state
                assert len(next_state.shape) == 3
                next_state = np.expand_dims(np.transpose(next_state, (2, 0, 1)), 0)

                reward_ += reward

                # Reward clipping and modification
                reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward

                # Store transition in memory
                memory.append((state, float(reward), int(action), next_state, int(1 - done)))
                state = next_state

                # Update Q-network if memory is sufficient
                if len(memory) > 2000:
                    loss += self.fit(memory)

                # Copy weights from Q1 to Q2
                if len(memory) > self.batch_size and len(memory) % 1 == 0:
                    self.Q2.load_state_dict(self.Q1.state_dict())

            # Update epsilon at the end of each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Stop timer
            end_time = time.time()
            elapsed_time = end_time - start_time

            with open(data_file, "a") as f:
                f.write("{}, {}, {:.2f}\n".format(episode, reward_, elapsed_time))

            print("Episode {}/{}, Reward: {:.2f}, Time: {:.2f}s".format(episode, self.episodes, reward_, elapsed_time))

            # Append rewards, times (i need this to plot metrics)
            self.episode_times.append(elapsed_time)
            self.rewards.append(reward_)

            reward_ = 0
