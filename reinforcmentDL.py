# Import necessary libraries
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython import display

# Create the CartPole environment with rendering enabled
env = gym.make("CartPole-v1", render_mode="human")

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a named tuple to store experience steps
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Define the replay memory to store past experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # Fixed-size queue

    def push(self, *args):
        # Save a transition to memory
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Randomly sample a batch of transitions
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the neural network for the Deep Q-Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        # Three fully connected layers
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Set hyperparameters
batch_size = 128
gamma = 0.99  # Discount factor
eps_start = 0.9  # Starting epsilon for exploration
eps_end = 0.05  # Minimum epsilon
eps_decay = 1000  # Epsilon decay rate
tau = 0.005  # Soft update parameter for target network
lr = 1e-4  # Learning rate

n_actions = env.action_space.n  # Number of possible actions

# Reset the environment to get initial state
state, info = env.reset()
n_observations = len(state)  # Number of inputs to the network

# Create the policy and target networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Copy weights from policy to target

# Set optimizer and replay memory
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = ReplayMemory(10000)

steps_done = 0  # Counter for total steps taken
episode_durations = []  # List to store episode lengths
show_result = False  # Flag for plot mode

# Function to select an action using epsilon-greedy policy
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1

    if sample > eps_threshold:
        # Exploit: choose the best known action
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Explore: choose a random action
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# Function to plot episode durations and moving average
def plot_durations():
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Plot moving average over 100 episodes
    if len(durations_t) > 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    display.display(plt.gcf())
    display.clear_output(wait=True)

# Function to optimize the model using one batch of experiences
def optimize_model():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Create mask and batch tensors for non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) for the actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for next states
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backpropagation and gradient clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()

# Run the main training loop
num_episodes = 250

for i_episode in range(num_episodes):
    # Reset environment and get initial state
    state, info = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

    for t in count():
        # Select and perform an action
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        # Convert observation to next state tensor
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        state = next_state

        # Perform one step of optimization
        optimize_model()

        # Soft update of target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key].clone().detach() * tau + target_net_state_dict[key] * (1 - tau)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

# Finalize training
print("Done")
show_result = True
plot_durations()
plt.ioff()
plt.show()
