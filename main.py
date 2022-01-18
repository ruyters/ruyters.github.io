# -*- coding: utf-8 -*-
from environment import Environment

# paramètres de l'environnement
env_name = 'CartPole-v1' # nom de l'environnement
episodes_nb_train = 500 # nombre d'épisodes entrainement
episodes_nb_perfs = 200 # nombre d'épisodes perfs
interactions_nb = 200 # nombre d'interactions par épisode

# paramètres sur les enregistrements et utilisations de performances
folder_name = 'new_test/perfs/3_double_nn_copy_method' # dossier
record = False # enregistrement des perfs
save_perfs = True # sauvegarde des performances des réseaux de neurones
load_perfs = False # chargement des performances des réseaux de neurones

# paramètres de l'agent
agent_memory = 1000 # mémoire du buffer circulaire
batch_exp_size = 32 # nombre d'expériences à récupérer de la mémoire
method_update_weights = 'scale' # méthode de mad des poids (copy | scale)

# création de l'environnement
env = Environment(
    env_name=env_name,
    agent_memory=agent_memory, 
    batch_exp_size=batch_exp_size, 
    method_update_weights=method_update_weights
)

# chargement des nn
env.load_perfs(
    folder_name=folder_name, 
    load_perfs=load_perfs
)

# lancement
env.run(
    episodes_nb=episodes_nb_train if record == False else episodes_nb_perfs, 
    interactions_nb=interactions_nb, 
    folder_name=folder_name,
    record=record
)

# sauvegarde des nn
env.save_perfs(
    folder_name=folder_name, 
    save_perfs=save_perfs
)