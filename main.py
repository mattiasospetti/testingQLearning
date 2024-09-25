# Testing Q-Learning algorithm.
# Source: https://www.geeksforgeeks.org/q-learning-in-python/

import numpy as np

n_states = 20  # Number of states in the grid world
n_actions = 4  # Number of possible actions (up, down, left, right)
goal_state = 19  # Goal state

# Initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

# Define parameters
learning_rate = 0.75
discount_factor = 0.90
exploration_prob = 0.2
epochs = 5000

# Q-learning algorithm
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states)  # Start from a random state

    while current_state != goal_state:
        # Choose action with epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit

        # Simulate the environment (move to the next state)
        # For simplicity, move to the next state
        next_state = (current_state + 1) % n_states

        # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
        reward = 1 if next_state == goal_state else 0

        # Update Q-value using the Q-learning update rule
        Q_table[current_state, action] += learning_rate * \
            (reward + discount_factor *
             np.max(Q_table[next_state]) - Q_table[current_state, action])

        current_state = next_state  # Move to the next state

# After training, the Q-table represents the learned Q-values
print("Learned Q-table:")
print(Q_table)


# ++ TO BE COMPLETED: ogni step ha lo stesso effetto, cioè si arriva allo stato successivo (e lo si conferma guardando la Q-Table finale).
# Inoltre si ha una reward solo alla fine --> penalità se sbaglio?