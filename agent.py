# -*- coding: utf-8 -*-
from collections import deque
import random
import numpy as np
from dqn import DQN
import torch

random.seed(42)

# paramètres
N = 10000 # nb de pas pour mettre à jour les poids par copie
# N = 1000 # nb de pas pour mettre à jour les poids par copie

class Agent:

    def __init__(self, memory_size, batch_exp_size, input_size, output_size, method_update_weights):
        self.memory = deque(maxlen=memory_size)
        self.batch_exp_size = batch_exp_size
        # réseaux de neurones
        self.policy_network = DQN(
            input_size=input_size, 
            output_size=output_size
        )
        self.target_network = DQN(
            input_size=input_size, 
            output_size=output_size
        )

        self.method_update_weights = method_update_weights
        self.pas = 0

    '''
        Mémorise une interaction au buffer circulaire
    '''
    def memorise_interaction(self, interaction):
        self.memory.append(interaction)

    '''
        Vérifie si on a assez d'interactions pour faire apprendre le NN
    '''
    def enough_interactions(self):
        return len(self.memory) >= self.batch_exp_size

    '''
        Retourne un ensemble d'interactions
    '''
    def get_batch_exp(self):
        if len(self.memory) == self.batch_exp_size: # pas assez d'expériences
            return np.array(self.memory, dtype = object)[0:len(self.memory)]
        else:
            random_number = random.randrange(len(self.memory) - self.batch_exp_size)
            return np.array(self.memory, dtype = object)[random_number:random_number + self.batch_exp_size]

    '''
        Fait apprendre le réseau de neurones
    '''
    def learn_nn(self):
        batch_exp = self.get_batch_exp()
        q_next_states_target = self.target_network.forward(torch.tensor(batch_exp[:, 2].tolist()))
        self.policy_network.learn(batch_exp=batch_exp, q_next_states=q_next_states_target)

    '''
        Applique une mise à jour des poids de Q' par Q
    '''
    def update_weights(self):
        self.pas += 1
        # print(self.pas)
        if self.method_update_weights == 'copy' and self.pas == N:
            self.pas = 0
            weights = self.policy_network.get_weights()
            # print("WEIGHTALORS", weights)
            self.target_network.update_weights_copy(weights)
        elif self.method_update_weights == 'scale':
            weights = self.policy_network.get_weights()
            self.target_network.update_weights_scale(weights)

    '''
        Récupère la meilleure action
    '''
    def best_action(self, state, method):
        if method == 'greedy':
            return self.policy_network.greedy_forward(state=state)
        else:
            return self.policy_network.basic_forward(state=state)

    '''
        Sauvegarde les configurations des nn dans un fichier
    '''
    def save_perfs(self, filename):
        torch.save({
                'policy_network': self.policy_network.state_dict(),
                'target_network': self.target_network.state_dict(),
            }, filename)

    '''
        Charge les configurations des nn dans un fichier
    '''
    def load_perfs(self, filename):
        checkpoint = torch.load(filename)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
    