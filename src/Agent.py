# this is a basic Q learning agent
# thants to yongwei xiao's article on Q-learning maze

import random
import numpy as np
import json
import pickle


class Agent():

    def __init__(self, player, epsilon=1, eps_dec=0.000001, eps_min=0.08, alpha = 0.1, gamma = 0.97):

        self.marker = player
        self.epsilon = epsilon
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_dec
        self.alpha = alpha
        self.gamma = gamma
        self.all_epsilon = []

        self.get_all_actions()

        self.Q_table = {} # tracks (state, action) tuples to track the value of an agent taking x in action in a state
        self.return_value = {} # tracks total return values for calculating the mean
        self.return_number = {} # tracks total returns for calculating the mean
        self.visited = [] # tracks visited states in an episode

    def save(self, iteration, path):

        dict_copy = dict()  # or {}
        for key, value in self.Q_table.items():
            dict_copy[str(key)] = self.Q_table[key]
        name = "player" + str(self.marker) + "_" + str(iteration)
        with open(path + "/" + name + "_Q_table.json", "w") as outfile:
            json.dump(dict_copy, outfile)
        with open(path + "/" + name + "_epsilon.txt", "wb") as fp:  # Pickling
            pickle.dump(self.all_epsilon, fp)


        # Updates the agent epsilon (called at the end of each episode)
    def update_epsilon(self):
        self.epsilon -= self.epsilon_dec
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def get_all_actions(self):
        self.all_actions = []
        for i in range(9):
            for j in range(9):
                self.all_actions.append((i, j))

    def select_action(self, state, legao_position) -> object:
        t_state = tuple(map(tuple, state))
        if t_state not in self.Q_table: # adding states to the table as agents come to them
            for move in self.all_actions:
                if (t_state, move) not in self.Q_table:
                    self.Q_table[(t_state, move)] = 0.05 # adding possible moves
                    self.return_value[(t_state, move)] = 0.0
                    self.return_number[(t_state, move)] = 0

        if random.random() > self.epsilon: # if choosing 'optimal' move
            action_scores = [self.Q_table[t_state, move] for move in legao_position] # get each legal action
            action_pos = np.argmax(action_scores) # get the best 'legal' action
            action = legao_position[action_pos] # find which action it relates to
        else: # if random move
            action = random.choice(legao_position)
        self.visited.append((t_state, action)) # store a list of 'visited' nodes for the episode
        return action


    def learn(self, winner):
        # self.marker is -1 for O player and 1 for X player
        # .. if O player gets reward of -1, can be inverted based on marker
        # .. if X player gets reward of -1, it will be negative

        # Episode reward
        reward = 0
        if self.marker == winner:
            reward = 1 # reward for a win
        elif self.marker == -winner: # episode was lost by current agent
            reward = -1 # reward for a loss
        else:
            reward = 0 # reward for a draw

        # Adding reward to return tables
        if reward != 0:
            for idx, (state, action) in enumerate(self.visited):
                G = 0
                discount = 1
                for t in range(idx):
                    G += reward * discount
                    discount *= 0.99
                    self.return_value[(state, action)] += G # add the return 'value'

        for idx, (state, action) in enumerate(self.visited):
            self.return_number[(state, action)] += 1 # state has been visited

        # Update states before the end
        for idx, (state, action) in enumerate(self.visited[:-1]):
            next_state, _ = self.visited[idx+1]
            max_Q = max([self.Q_table[(next_state, a)] for a in self.all_actions])
            self.Q_table[(state, action)] = 0.9 * self.Q_table[(state, action)] + 0.1 * (reward + 0.97 * max_Q - self.Q_table[(state, action)])

        # Update the last state
        (last_state, last_action) = self.visited[-1]  # check update
        self.Q_table[(last_state, last_action)] = 0.9 * self.Q_table[(last_state, last_action)] + 0.1 * (reward - self.Q_table[(last_state, last_action)])

        # clearing states visited for the episode
        self.visited = []
