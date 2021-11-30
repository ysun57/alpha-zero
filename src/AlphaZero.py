# thanks to Arnav Paruthi's clear explaination about Alpha Zero

from keras import layers
from keras.models import Model
from keras import models
from keras.optimizers import Adam
from keras.models import load_model

import numpy as np
import math
import os
from datetime import datetime
from copy import deepcopy

from src.Board import Board

class AlphaZero():

    def __init__(self):
        # hyperparameters
        self.train_episodes = 2
        self.mcts_search = 600
        self.n_competation_network = 50
        self.play_games_before_training = 2
        self.cpuct = 4
        self.training_epochs = 2
        self.learning_rate = 0.001
        self.save_model_path = 'training_record'
        self.learning_rate = 0.001
        self.win_thresh = 0.52

        # initializing MC search tree
        self.Q = {}  # state
        self.Nsa = {}  # number of times certain state-action pair has been visited
        self.Ns = {}  # number of times state has been visited
        self.W = {}  # number of total points collected after taking state action pair
        self.P = {}  # initial predicted probabilities of taking certain actions in state

        self.nn = self.create_neural_network()
        self.old_nn = self.nn

        self.board = Board()
        self.virtual_board = Board()
        self.training_history = []

    def create_neural_network(self):
        input_layer = layers.Input(shape=(9, 9), name="BoardInput")
        reshape = layers.core.Reshape((9, 9, 1))(input_layer)
        conv_1 = layers.Conv2D(128, (3, 3), padding='valid', activation='relu', name='conv1')(reshape)
        conv_2 = layers.Conv2D(128, (3, 3), padding='valid', activation='relu', name='conv2')(conv_1)
        conv_3 = layers.Conv2D(128, (3, 3), padding='valid', activation='relu', name='conv3')(conv_2)

        conv_3_flat = layers.Flatten()(conv_3)

        dense_1 = layers.Dense(512, activation='relu', name='dense1')(conv_3_flat)
        dense_2 = layers.Dense(256, activation='relu', name='dense2')(dense_1)

        pi = layers.Dense(81, activation="softmax", name='pi')(dense_2)
        v = layers.Dense(1, activation="tanh", name='value')(dense_2)

        model = Model(inputs=input_layer, outputs=[pi, v])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.learning_rate))

        model.summary()
        return model

    def train_nn(self, game_record):

        state = []
        policy = []
        value = []

        for record in game_record:
            state.append(record[0])
            policy.append(record[2])
            value.append(record[3])

        state = np.array(state)
        policy = np.array(policy)
        value = np.array(value)

        history = self.nn.fit(state, [policy, value], batch_size=32, epochs=self.training_epochs, verbose=1)
        self.training_history.append(history)

    # mc tree search
    # two board inside, only update the visual board.
    def mcts(self, current_player):

        valid_point = self.virtual_board.check_valid_position()

        sTuple = tuple(map(tuple, self.virtual_board.board))

        if len(valid_point) > 0:
            if sTuple not in self.P.keys():

                policy, v = self.nn.predict(self.virtual_board.board.reshape(1, 9, 9))
                v = v[0][0]
                valids = np.zeros(81)
                for possible_a in valid_point:
                    valids[possible_a[0]*9 + possible_a[1]] = 1
                policy = policy.reshape(81) * valids
                policy = policy / np.sum(policy)
                self.P[sTuple] = policy

                self.Ns[sTuple] = 1

                for a in valid_point:
                    self.Q[(sTuple, a)] = 0
                    self.Nsa[(sTuple, a)] = 0
                    self.W[(sTuple, a)] = 0
                return -v

            best_uct = -100
            for a in valid_point:
                try:
                    if (sTuple, a) not in self.Q.keys():
                        self.Q[(sTuple, a)] = 0
                        self.Nsa[(sTuple, a)] = 0
                        self.W[(sTuple, a)] = 0
                    uct_a = self.Q[(sTuple, a)] + self.cpuct * self.P[sTuple][a[0]*9 + a[1]] * \
                        (math.sqrt(self.Ns[sTuple]) / (1 + self.Nsa[(sTuple, a)]))
                except:
                    # debug use, remove soon
                    print("strop here")


                if uct_a > best_uct:
                    best_uct = uct_a
                    best_a = a

            self.virtual_board.place(best_a, current_player)

            if self.virtual_board.get_winner():
                v = current_player
            else:
                current_player *= -1
                v = self.mcts(current_player)
        else:
            return 0

        self.W[(sTuple, best_a)] += v
        self.Ns[sTuple] += 1
        self.Nsa[(sTuple, best_a)] += 1
        self.Q[(sTuple, best_a)] = self.W[(sTuple, best_a)] / self.Nsa[(sTuple, best_a)]
        return -v

    # get the action probs. Useful for getting policy
    def calculate_action_prob(self, current_player):

        for _ in range(self.mcts_search):
            self.virtual_board = deepcopy(self.board)
            value = self.mcts(current_player)

        print("finish MCTS!")

        actions_dict = {}

        sTuple = tuple(map(tuple, self.board.board))

        for a in self.board.check_valid_position():
            actions_dict[a] = self.Nsa[(sTuple, a)] / self.Ns[sTuple]

        action_probs = np.zeros(81)

        for a in actions_dict:
            np.put(action_probs, a[0] * 9 + a[1], actions_dict[a], mode='raise')

        return action_probs

    # play with the agent itself
    def play_game(self):

        current_player = 1
        game_record = []

        self.board.reset()

        while self.board.check_continue():
            policy = self.calculate_action_prob(current_player)
            policy = policy / np.sum(policy)
            game_record.append([self.board.board, current_player, policy, None])
            action = np.random.choice(len(policy), p=policy)

            print("alpha zero chosen action", action)
            print(self.board.see_board)

            action_position = (action // 9, action % 9)

            self.board.place(action_position, current_player)

            if len(self.board.check_valid_position()) == 0:
                for tup in game_record:
                    tup[3] = 0
                return game_record

            if not self.board.check_continue():
                for tup in game_record:
                    if tup[1] == current_player:
                        tup[3] = 1
                    else:
                        tup[3] = -1
                return game_record

            current_player *= -1

    # let old one battle the new one, and get the best
    def competation(self):

        nn_wins = 0
        new_nn_wins = 0

        for _ in range(self.n_competation_network):

            s = []

            for _ in range(9):
                s.append(np.zeros(9))

            # cannot use condition here because there are two players
            while True:

                policy, v = self.nn.predict(self.board.board.reshape(1, 9, 9))
                valids = np.zeros(81)

                valid_point = self.board.check_valid_position()

                if len(valid_point) == 0:
                    break

                np.put(valids, valid_point, 1)
                policy = policy.reshape(81) * valids
                policy = policy / np.sum(policy)
                action = np.argmax(policy)
                action_position = (action // 9, action % 9)
                self.board.place(action_position, 1)

                if self.board.get_winner() == 1:
                    nn_wins += 1
                    break

                policy, v = self.nn.predict(self.board.board.reshape(1, 9, 9))
                valids = np.zeros(81)

                valid_point = self.board.check_valid_position()

                if len(valid_point) == 0:
                    break

                np.put(valids, valid_point, 1)
                policy = policy.reshape(81) * valids
                policy = policy / np.sum(policy)
                action = np.argmax(policy)
                action_position = (action // 9, action % 9)

                self.board.place(action_position, 1)

                if self.board.get_winner() == -1:
                    new_nn_wins += 1
                    break

        if (new_nn_wins + nn_wins) == 0:
            now = datetime.utcnow()
            filename = 'ultimate_tic_tac_toe{}.h5'.format(now)
            model_path = os.path.join(self.save_model_path, filename)
            self.nn.save(model_path)
            return False

        win_percent = float(new_nn_wins) / float(new_nn_wins + nn_wins)
        if win_percent > self.win_thresh:
            print("new network win")
            print(win_percent)
            return True
        else:
            print("old network win")
            print(new_nn_wins)
            return False

    def train(self):

        game_record = []

        for episode in range(self.train_episodes):

            self.nn.save('temp.h5')
            self.old_nn = models.load_model('temp.h5')

            for _ in range(self.play_games_before_training):
                game_record += self.play_game()

            self.train_nn(game_record)
            game_record = []
            if self.competation():
                self.old_nn = self.nn
                self.Q = {}
                self.Nsa = {}
                self.Ns = {}
                self.W = {}
                self.P = {}
            else:
                self.nn = self.old_nn

        now = datetime.utcnow()
        filename = 'ultimate_tic_tac_toe_tree_size_600{}.h5'.format(now)
        model_path = os.path.join(self.save_model_path, filename)
        self.nn.save(str(model_path))

    # load from existing network. Then no need to retrain
    def load_network(self):
        self.nn = load_model("temp.h5")
        print("load finished!")

    # interface with playWithAlphazeroPlayer 1 and 2
    def play_with_human(self, board, last_position):
        self.board.board = board
        self.board.last_position = last_position
        policy = self.calculate_action_prob(-1)
        valid = self.board.check_valid_position()
        valid_policy = []
        for a in valid:
            valid_policy.append(policy[a[0] * 9 + a[1]])
        valid_policy = valid_policy / np.sum(valid_policy)
        action = np.argmax(valid_policy)
        return valid[action]

def test():
    alphazero = AlphaZero()
    alphazero.train()

if __name__ == '__main__':
    test()