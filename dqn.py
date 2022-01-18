# -*- coding: utf-8 -*-
import torch
import random

from torch.nn.modules.activation import Sigmoid

random.seed(42)

# paramètres du réseau de neurones
hidden_size = 12
eps_max = 0.99
eps_min = 0.02
# eps_min = 0.5
beta = 0.995 
gamma = 0.999 # taux de rabais (rendement), généralement compris entre 0.95 et 0.99 
# gamma = 0.95

eta = 1e-2
alpha = 5e-2
uniform_weights = 1e-3
reduction='mean' # méthode de réduction de la loss func (mean | sum)

class DQN(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.input_size = input_size # taille d'entrée
        self.output_size = output_size # taille de sortie

        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

        torch.nn.init.uniform_(self.hidden.weight,-0.001,0.001)
        torch.nn.init.uniform_(self.output.weight,-0.1,0.1)

        # torch.nn.init.normal_(self.hidden.weight, mean=0, std=1)
        # torch.nn.init.normal_(self.output.weight, mean=0, std=1)
        
        self.loss_function = torch.nn.MSELoss(reduction=reduction) # fonction d'erreur
        # self.loss_function = torch.nn.BCELoss() # fonction d'erreur
         #self.optim = torch.optim.SGD(self.parameters(), lr=eta) # descente de gradient
        self.optim = torch.optim.Adam(self.parameters(), lr=eta) # descente de gradient
        # self.optim = torch.optim.RMSprop(self.parameters(), lr=eta) # descente de gradient

        # paramètres
        self.eps = eps_max


    '''
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.input_size = input_size # taille d'entrée
        self.output_size = output_size # taille de sortie

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size),
            # torch.nn.ReLU(),
            # Sigmoid(), # meilleur avec sigmoid
            torch.nn.Linear(hidden_size, self.output_size)
        )

        torch.nn.init.normal_(self.layers[0].weight, mean=0, std=1)
        torch.nn.init.normal_(self.layers[1].weight, mean=0, std=1)
        # torch.nn.init.uniform_(self.layers[0].weight,-0.1,0.1)
        # torch.nn.init.uniform_(self.layers[1].weight,-0.1,0.1)
        '''

        

    def get_weights(self):
        weights = []
        #f or layer in self.layers:
           # weights.append(layer.weight)
        weights.append(self.hidden.weight)
        weights.append(self.output.weight)
        return weights

    '''
        Met à jour les poids par copie intégrale
    '''
    def update_weights_copy(self, weights):
        # for layer, index in self.layers:
        # print(self.layer)
        self.hidden.weight = weights[0]
        self.output.weight = weights[1]

    '''
        Met à jour les poids à l'aide d'un scalaire
    '''
    def update_weights_scale(self, weights):
        # for layer, index in self.layers:
            #layer.weight = torch.nn.Parameter((1 - alpha) * layer.weight + alpha * weights[index])
            # layer.weight = torch.nn.Parameter(torch.add(torch.mul((1 - alpha), layer.weight), torch.mul(alpha, weights[index])))
        self.hidden.weight = torch.nn.Parameter(torch.add(torch.mul((1 - alpha), self.hidden.weight), torch.mul(alpha, weights[0])))
        self.output.weight = torch.nn.Parameter(torch.add(torch.mul((1 - alpha), self.output.weight), torch.mul(alpha, weights[1])))

    '''
        Calcul des Q_val pour chaque action
    '''
    # def forward(self, state):
        # return self.layers(state)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

    '''
        Calcul des Q_val pour chaque action (sans exploration greedy)
        Retourne la meilleure action
    '''
    def basic_forward(self, state):
        state = torch.tensor(state)
        actions = self.forward(state)
        return torch.argmax(actions).item()

    '''
        Calcul des Q_val pour chaque action (avec exploration greedy)
        Retourne la meilleure action OU action aléatoire
    '''
    def greedy_forward(self, state):
        random_value = random.uniform(0, 1)
        self.eps = beta * self.eps if beta * self.eps > eps_min else self.eps
        if random_value < self.eps:
            return random.randint(0, self.output_size - 1)
        else:
            return self.basic_forward(state)

    '''
        Apprend, calcule la fct d'erreur et fait une descente de gradient
    '''
    def learn(self, batch_exp, q_next_states):
        states = torch.tensor(batch_exp[:, 0].tolist())
        actions = torch.tensor(batch_exp[:, 1].tolist())
        #next_states = torch.tensor(batch_exp[:, 2].tolist())
        rewards = torch.tensor(batch_exp[:, 3].tolist())
        not_done = torch.tensor(batch_exp[:, 4].tolist())

        # q_valeurs pour les états et états suivants
        q_states = self.forward(states)

        # prédiction pour calculer les q-valeurs
        predictions = torch.flatten(torch.gather(q_states, 1, torch.reshape(actions, (1, len(batch_exp)))))
        #print(predictions)
        
        # cible
        _, best_q_next_states = torch.max(q_next_states, dim=1) # meilleure q_valeur de l'action de notre prochain état
        # ("Q-STATES", q_states)
        # print("BEST", best_q_next_states)
        targets = torch.add(rewards, torch.mul(not_done, torch.mul(gamma, best_q_next_states)))

        # fct cout total + rétropropagation
        loss = self.loss_function(targets, predictions)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        
        
        



