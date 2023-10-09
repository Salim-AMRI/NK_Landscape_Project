# Importation de bibliotheque
import numpy as np
import argparse
import datetime
import torch
import cma
import os
from joblib import Parallel, delayed
#from tqdm import tqdm
import random
import logging

# Importation de modules personnalisés

from Env_nk_landscape import EnvNKlandscape
from Instances_Generator import Nk_generator
import torch.nn.functional as F


# Création du parser d'arguments
from strategies.HillClimber import HillClimber
from strategies.HillClimberSoftmax import HillClimberSoftmax
from strategies.IteratedHillClimber import IteratedHillClimber
from strategies.StrategyNN import StrategyNN
from strategies.StrategyNNTabu import StrategyNNTabu
from strategies.StrategyNNHistoryTabu import StrategyNNHistoryTabu
from strategies.strategyNNLastMoveIndicatorTabu import StrategyNNLastMoveIndicatorTabu
from strategies.Tabu import Tabu

def get_Score_trajectory(strategy, N, K, path, nb_intances, idx_run, alpha=None, withLogs = False):


    # Déterminez le numéro d'instance et de redémarrage
    num_instance = idx_run % nb_intances
    num_restart = idx_run // nb_intances

    if(withLogs):
        # Définissez le chemin du répertoire de journalisation
        log_directory = 'log_trajectory'

        # Assurez-vous que le répertoire de journalisation existe, sinon créez-le
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        # Créez un nom de fichier de journal personnalisé en fonction des paramètres
        log_filename = f"logTrajectory_{strategy.toString()}_N_{N}_K_{K}_nb_instances_test_{num_instance}_nb_restarts_{num_restart}.log"
        print(log_filename)

        log_file = os.path.join(log_directory, log_filename)

        # Configurez la journalisation pour écrire les fichiers de journalisation dans le répertoire spécifié
        logging.basicConfig(filename=log_file, level=logging.INFO)


    name_instance = path + "nk_" + str(N) + "_" + str(K) + "_" + str(num_instance) + ".txt"

    strategy.reset()

    #print("launch instance " + name_instance)
    env = EnvNKlandscape(name_instance, 2*N)
    env.reset()
    terminated = False
    current_score = env.score()
    bestScore = current_score



    while not terminated:

        action_id = strategy.choose_action(env)

        if action_id >= 0:
            state, deltaScore, terminated = env.step(action_id)
            current_score += deltaScore

        # action de réaliser une perturbation
        elif action_id == -1:

            env.perturbation(alpha)
            current_score = env.score()

        strategy.updateInfo(action_id, env.turn)

        if current_score > bestScore:
            bestScore = current_score

        if withLogs:

            # Vous pouvez ajouter le suivi du nombre d'actions positives et supérieures ici
            # Suivre le nombre d'actions positives
            positive_count = sum(1 for action in strategy.neighDeltaFitness if action > 0)

            # Suivre le nombre d'actions supérieures à celle choisie par le réseau
            actions_above_count = sum(1 for action in strategy.neighDeltaFitness if action > deltaScore)

            # Ajoutez cet appel pour enregistrer le score, l'action_id et sa contribution dans le fichier journal à chaque itération
            if withLogs:
                logging.info(
                    f"Turn: {env.turn}, Current Score: {current_score}, Action_id: {action_id}, Action Contribution: {deltaScore}, Positive Actions: {positive_count}, Actions Above Chosen: {actions_above_count}")


    return bestScore




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Optimisation de poids de réseau de neurones avec CMA-ES')

    # Ajout des arguments
    parser.add_argument('type_strategy', type=str, help='type_strategy')
    parser.add_argument('N', type=int, help='Taille de l\'instance')
    parser.add_argument('K', type=int, help='Paramètre K')
    parser.add_argument('--hiddenlayer_size', nargs='+', type=int, default=[10,5], help='Hidden layer sizes')
    parser.add_argument('--nb_restarts', type=int, default=5, help='Nombre de redémarrages')
    parser.add_argument('--nb_instances', type=int, default=10, help='Nombre d\'instances')
    parser.add_argument('--nb_jobs', type=int, default=-1, help='Nombre de jobs')
    parser.add_argument('--sigma_init', type=float, default=0.2, help='Ecart-type initial')
    parser.add_argument('--alpha', type=float, default=0.1, help='Nombre de bits perturbées')
    parser.add_argument('--max_generations', type=int, default=10000, help='Nombre de générations')
    parser.add_argument('--verbose', action='store_true', help='Afficher des informations de progression')
    parser.add_argument('--seed', type=int, default=0, help='Seed pour la génération aléatoire')
    parser.add_argument('--use_trainset',  default=False, action='store_true')

    # Analyse des arguments de ligne de commande
    args = parser.parse_args()

    # Paramètres initiaux en fonction des arguments
    N = args.N
    K = args.K
    seed = args.seed
    alpha = args.alpha
    sigma_init = args.sigma_init
    nb_restarts = args.nb_restarts
    nb_instances = args.nb_instances
    type_strategy = args.type_strategy
    max_generations = args.max_generations
    nb_jobs = args.nb_jobs
    hiddenlayer_size = args.hiddenlayer_size

    print(" test hiddenlayer_size")
    print(hiddenlayer_size)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if(nb_jobs == -1):
        nb_jobs = nb_instances*nb_restarts



    if(args.use_trainset):
        train_path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/train/"
    else:
        train_path = "./tmp/" + type_strategy + "_seed" + str(seed) + "/"

    valid_path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/validation/"
    pathResult = "results/"


    if not os.path.exists("solutions"):
        os.makedirs("solutions")
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    str_ = ""
    for idx, layer in enumerate(hiddenlayer_size):
        if(idx == len(hiddenlayer_size) - 1):
            str_ += str(layer)
        else:
            str_ += str(layer) + ","


    # Utilisez datetime.datetime.now() pour obtenir la date actuelle
    nameResult = "test_strategy_" + type_strategy + "_" + str_ + "_N_" + str(N) + "_K_" + str(K) + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(seed) + ".txt"

    print(nameResult)

    f = open(os.path.join(pathResult, nameResult), "w")
    f.write("generation,avg_training_score,avg_validation_score\n")
    f.close()

    hillClimber_results = []
    IteratedhillClimber_results = []
    tabou_results = []





    ## Add save result
    if "NN" in type_strategy:



        if(type_strategy == "strategyNN"):
            list_strategy = [StrategyNN(N, hiddenlayer_size ) for idx_run in range(nb_instances * nb_restarts)]

        # elif(type_strategy == "strategyNN_Tabu"):
        #     tabuTime = 3 + int(0.1 * N)
        #     list_strategy = [StrategyNN_Tabu(N, hiddenlayer_size, tabuTime) for idx_run in range(nb_instances * nb_restarts)]

        # elif (type_strategy == "strategyNN_historyTabu"):
        #     sizeHistory = N//2
        #     list_strategy = [StrategyNN_historyTabu(N, hidden_layers_size, sizeHistory) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "strategyNNLastMoveIndicatorTabu"):
            list_strategy = [StrategyNNLastMoveIndicatorTabu(N, hiddenlayer_size ) for idx_run in range(nb_instances * nb_restarts)]


        # Initialisation de CMA-ES avec un vecteur de poids initial aléatoire
        initial_solution = np.random.randn(list_strategy[0].num_params)
        es = cma.CMAEvolutionStrategy(initial_solution, sigma_init, {'seed':seed})

        print("Taille de la population dans CMA-ES :", es.popsize)

        best_global_validation_score = float("-inf")

        for generation in range(max_generations):


            if(args.use_trainset == False):
                print("start instances generation")
                # Générer de nouvelles instances de formation à chaque itération
                Nk_generator(N, K, nb_instances, train_path)
                print("end instances generation")

            solutions = es.ask()  # Échantillonnez de nouveaux vecteurs de poids

            # Évaluez les performances de chaque solution en parallèle
            list_training_scores = []

            best_current_score = float("-inf")

            for idx, solution in enumerate(solutions):
                print("Eval individu " + str(idx))


                for idx_run in range(nb_instances * nb_restarts):
                    list_strategy[idx_run].update_weights(solution)

                list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[idx_run], N, K, train_path, nb_instances, idx_run) for idx_run
                    in range(nb_instances * nb_restarts))

                average_training_score = np.mean(list_scores)

                list_training_scores.append(average_training_score)

                if average_training_score > best_current_score:

                    best_current_score = average_training_score
                    best_current_solution = np.copy(solution)


            # Mettez à jour CMA-ES avec les performances
            es.tell(solutions, -np.array(list_training_scores))

            list_strategy[0].update_weights(best_current_solution)

            # Évaluez la meilleure solution sur l'ensemble de validation
            list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[0], N, K, valid_path, nb_instances, idx_run) for idx_run  in range(nb_instances * nb_restarts))

            average_validation_score = np.mean(list_scores)

            print("Score moyen sur l'ensemble de validation : " + str(average_validation_score))

            f = open(pathResult + nameResult, "a")
            f.write(str(generation) + "," + str(max(list_training_scores)) + "," + str(average_validation_score) + "\n")
            f.close()

            if(average_validation_score > best_global_validation_score):
                best_global_validation_score = average_validation_score
                np.savetxt("solutions/best_solution_" + nameResult + ".csv" , best_current_solution)



    elif(type_strategy == "tabu"):

        best_current_score = float("-inf")

        if (args.use_trainset == False):
            print("start instances generation")
            # Générer de nouvelles instances de formation à chaque itération
            Nk_generator(N, K, nb_instances, train_path)
            print("end instances generation")

        for tabuTime in range(N):

            list_strategy = [Tabu(N, tabuTime) for idx_run in range(nb_instances * nb_restarts)]

            # Évaluez les performances de chaque solution en parallèle
            list_training_scores = []


            list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[idx_run], N, K, train_path, nb_instances, idx_run) for idx_run
                    in range(nb_instances * nb_restarts))


            average_training_score = np.mean(list_scores)
            print("tabuTime : " + str(tabuTime) + " - average_training_score : " + str(average_training_score))

            if average_training_score > best_current_score:
                best_current_score = average_training_score
                best_tabuTime = tabuTime

        list_strategy = [Tabu(N, best_tabuTime) for idx_run in range(nb_instances * nb_restarts)]
        # Évaluez la meilleure solution sur l'ensemble de validation
        list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[0], N, K, valid_path, nb_instances, idx_run) for idx_run  in range(nb_instances * nb_restarts))

        average_validation_score = np.mean(list_scores)

        print("Score moyen sur l'ensemble de validation : " + str(average_validation_score))
        print("best_tabuTime : " + str(best_tabuTime))

        f = open(pathResult + nameResult + "_best_tabuTime_" + str(best_tabuTime), "a")
        f.write(str(best_tabuTime) + "," + str(average_training_score) + "," + str(average_validation_score) + "\n")
        f.close()



    else:


        print("Évaluation de la stratégie " + type_strategy)

        # if(type_strategy == "hillClimber"):
        list_strategy = [HillClimber(N) for idx_run in range(nb_instances * nb_restarts)]
        # elif(type_strategy == "hillClimberSoftmax"):
        #     list_strategy = [HillClimberSoftmax(N) for idx_run in range(nb_instances * nb_restarts)]
        # elif (type_strategy == "iteratedHillClimber"):
        #     list_strategy = [IteratedHillClimber(N) for idx_run in range(nb_instances * nb_restarts)]

        list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[idx_run], N, K, valid_path, nb_instances, idx_run, alpha=alpha) for idx_run in
            range(nb_instances * nb_restarts))

        average_score_baseline = np.mean(list_scores)

        print("Score moyen de la stratégie " + type_strategy + " sur l'ensemble de validation :")
        print(average_score_baseline)
        f = open(pathResult + nameResult, "a")
        f.write(str(0) + ",," + str(average_score_baseline) + "\n")
        f.close()

