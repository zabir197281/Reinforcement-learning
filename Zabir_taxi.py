import numpy as np
import gym
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create the environment
env = gym.make("Taxi-v3", render_mode='ansi')

# Initialize the Q-table to store Q-values for state-action pairs
Q_reward = np.zeros((env.observation_space.n, env.action_space.n))

# Training parameters for Q-learning
alpha = 0.9  # Learning rate
gamma = 0.9  # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500  # Maximum steps per episode
epsilon = 0.1  # Epsilon-greedy exploration probability

print("\nStarted training the Q-learning model . . .\n")

# Training the Q-learning agent
for episode in range(num_of_episodes):
    state = env.reset()[0]

    for step in range(num_of_steps):
        # Choose an action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(Q_reward[state, :])  # Exploit learned values

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _, _ = env.step(action)

        # Update the Q-value for the state-action pair using the Q-learning update formula
        old_value = Q_reward[state, action]
        next_max = np.max(Q_reward[next_state, :])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q_reward[state, action] = new_value

        # Move to the next state
        state = next_state

# Testing the trained Q-learning agent
print("Testing the trained agent:\n")

list_of_rewards = []  # To store the total rewards for each testing run
list_of_actions = []  # To store the total number of actions for each testing run

for test_run in range(10):
    state = env.reset()[0]
    total_reward = 0
    num_of_actions = 0

    for t in range(50):  # Maximum of 50 steps per testing episode
        action = np.argmax(Q_reward[state, :])  # Exploit learned values
        state, reward, done, truncated, info = env.step(action)

        total_reward += reward
        num_of_actions += 1
        print(env.render())  # Display the environment
        time.sleep(1)  # Pause for visualization

        if done:
            print("===========================\n")
            print(f"For {test_run+1}th testing run, the total reward is: {total_reward}")
            print("===========================\n")
            list_of_rewards.append(total_reward)
            list_of_actions.append(num_of_actions)
            break

env.close()

# Close the environment rendering

# Calculate and display the average total reward and average number of actions over the testing runs
average_total_reward = sum(list_of_rewards) / 10
average_num_actions = sum(list_of_actions) / 10

print("Average Total Reward:", average_total_reward)
print("Average Number of Actions:", average_num_actions)

print()

warnings.filterwarnings("default", category=DeprecationWarning)
