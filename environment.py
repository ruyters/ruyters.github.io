# -*- coding: utf-8 -*-
import gym
from matplotlib import pyplot as plt
from agent import Agent

# nom de fichiers
filename_vid = 'perfs.mp4'
filename_perfs = 'perfs.png'
filename_training = 'train.png'
filename_config = 'config.pth'

class Environment:

    def __init__(self, env_name, agent_memory, batch_exp_size, method_update_weights):
        self.env = gym.make(env_name) # création de l'environnement
        self.env.seed(1)

        # pour les résultats graphiques
        self.total_interactions = [0]
        self.total_rewards = [0]

        self.agent = Agent(
            memory_size=agent_memory, 
            batch_exp_size=batch_exp_size, 
            input_size=self.env.observation_space.shape[0],  # input_size= 4, 
            output_size=self.env.action_space.n, 
            # output_size= 1 , 
            method_update_weights=method_update_weights
        )


    '''
        Démarre un épisode
    '''
    def start_episode(self, interactions_nb, video_recorder):
        observation = self.env.reset()
        rewards_episode = 0
        interactions_episode = 0

        # lancement des interactions
        print("###", interactions_nb)
        for t in range(interactions_nb):
            print("Episode numéro : ", t)
            self.env.render() # affichage de l'environnement

            if video_recorder is not None:
                video_recorder.capture_frame()

            state = observation # état actuel

            # exécution d'une action
            action = self.agent.best_action(state=state, method='greedy' if video_recorder is None else 'best')
            observation, reward, done, _ = self.env.step(action)

            next_state = observation # état suivant

            # mémorisation de l'interaction si on est en apprentissage
            # on inverse le booléen de done pour la suite (si l'épisode continue => true, sinon false)
            interaction = (state.tolist(), action, next_state.tolist(), reward, not done)
            self.agent.memorise_interaction(interaction)

            if self.agent.enough_interactions():
                self.agent.learn_nn()
                self.agent.update_weights()

            
            if done:
                if interactions_episode < 30:
                    reward = -5 
                if (interactions_episode < 50 and interactions_episode>30):
                        reward = -1 
                else:
                    reward = 5
            
            rewards_episode += reward
            interactions_episode += 1

            # vérification si fin de l'épisode
            if done or t == interactions_nb - 1:
                self.total_rewards.append(rewards_episode)
                self.total_interactions.append(self.total_interactions[-1] + interactions_episode)
                print(f'Episode finished after {interactions_episode} interactions')
                break


    '''
        Enregistre les performances (mode apprentissage et mode record)
    '''
    def run(self, episodes_nb, interactions_nb, folder_name, record):
        # lancement du record si fichier spécifié
        if record == False:
            video_recorder = None
        else:
            filename = folder_name + '/' + filename_vid
            video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(
                env=self.env, 
                path=filename
            )
        
        # lancement des épisodes
        for i_episode in range(episodes_nb):
            self.start_episode(
                interactions_nb=interactions_nb, 
                video_recorder=video_recorder
            )

        # fermeture de l'environnement
        self.env.close()

        # fermeture de la vidéo
        if record == True:
            print(f'Sauvegarde de la vidéo dans le fichier {filename}')
            video_recorder.close()

        # affichage graphique (ou sauvegarde)
        self.plot_experiments(
            episodes_nb=episodes_nb, 
            interactions_nb=interactions_nb, 
            folder_name=folder_name,
            record=record
        )

    '''
        Représentation graphique / Sauvegarde du graphique
    '''
    def plot_experiments(self, episodes_nb, interactions_nb, folder_name, record):
        # affichage graphique, enregistrement
        plt.plot(self.total_interactions, self.total_rewards)

        # trait vertical pour marquer la fin d'un épisode
        if record == True:
            for episode in self.total_interactions:
                plt.axvline(x=episode, c = 'red', linestyle = '--', alpha = .3)

        # limites verticales/horizontales
        plt.xlim(0, self.total_interactions[-1])
        plt.ylim(0, interactions_nb + 1)

        # légendes
        plt.xlabel(f'nb interactions selon épisodes ({episodes_nb} épisodes)')
        plt.ylabel(f'récompense total (récompense max: {max(self.total_rewards)})')

        # on sauvegarde la config si le nom du fichier existe, sinon on show
        if record == True:
            # mode training
            filename = folder_name + '/' + filename_perfs
            print(f'Tentative de sauvegarde de l\'entrainement dans le fichier {filename}..')
        else:
            # mode perfs
            filename = folder_name + '/' + filename_training
            print(f'Tentative de sauvegarde des performances dans le fichier {filename}..')
        plt.savefig(filename)
        print(f'Sauvegarde réussie!')

    '''
        Sauvegarde les configurations des nn dans un fichier
    '''
    def save_perfs(self, folder_name, save_perfs):
        if save_perfs == False:
            return
        filename = folder_name + '/' + filename_config
        self.agent.save_perfs(filename=filename)
        print(f'Sauvegarde des réseaux de neurones dans le fichier {filename}')

    '''
        Charge les configurations des nn dans un fichier
    '''
    def load_perfs(self, folder_name, load_perfs):
        if load_perfs == False:
            return
        filename = folder_name + '/' + filename_config
        self.agent.load_perfs(filename=filename)
        print(f'Chargement des réseaux de neurones dans le fichier {filename}')