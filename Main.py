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


# Importation de modules personnalisés
from EnvUBQPLandscape import EnvUBQPLandscape
from Env_nk_landscape import EnvNKlandscape
from Instances_Generator import Nk_generator

# Création du parser d'arguments
from strategies.EmergingDeterministicStrategyK8 import EmergingDeterministicStrategyK8
from strategies.HillClimber import HillClimber
from strategies.HillClimberFirstImprovementJump import HillClimberFirstImprovementJump
from strategies.HillClimberJump import HillClimberJump
from strategies.IteratedHillClimber import IteratedHillClimber
from strategies.OneLambdaDeterministic import OneLambdaDeterministic
from strategies.StrategyNN import StrategyNN
from strategies.strategyNNLastMoveIndicatorTabu import StrategyNNLastMoveIndicatorTabu
from strategies.Tabu import Tabu

from strategies.StrategyNNRanked_v0 import StrategyNNRanked_v0
from strategies.StrategyNNRanked_v1 import StrategyNNRanked_v1
from strategies.StrategyNNRanked_v2 import StrategyNNRanked_v2

#from strategies.StrategyNNRanked_v1_zScore import StrategyNNRanked_v1_zScore
#from strategies.StrategyNNRanked_v2_zScore import StrategyNNRanked_v2_zScore

from strategies.StrategyNNFitness import StrategyNNFitness

from strategies.StrategyNNFitness_and_current import StrategyNNFitness_and_current

from strategies.StrategyNNRanked_delta_rescale import StrategyNNRanked_delta_rescale


def get_Score_trajectory(type_problem, strategy, N, K, path, nb_intances, idx_run, alpha=None, withLogs = False, starting_point = None):

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

        input_filename = f"input_{strategy.toString()}_N_{N}_K_{K}_nb_instances_test_{num_instance}_nb_restarts_{num_restart}.log"

        output_filename = f"output_{strategy.toString()}_N_{N}_K_{K}_nb_instances_test_{num_instance}_nb_restarts_{num_restart}.log"


        log_file = os.path.join(log_directory, log_filename)

        input_file = os.path.join(log_directory, input_filename)
        output_file = os.path.join(log_directory, output_filename)

        log_file_trajectory = open(log_file, "w")
        input_file_trajectory = open(input_file, "w")
        output_file_trajectory = open(output_file, "w")







    strategy.reset()

    #print("launch instance " + name_instance)

    if(type_problem == "NK"):
        name_instance = path + "nk_" + str(N) + "_" + str(K) + "_" + str(num_instance) + ".txt"
        env = EnvNKlandscape(name_instance, 2*N)
    elif(type_problem == "UBQP"):
        name_instance = path + "puboi_evo_n_" + str(N) + "_t_" + str(K) + "_i_" + str(1001 + num_instance) + ".json"

        env = EnvUBQPLandscape(name_instance, 2*N)


    if(starting_point is None):
        env.reset()
    else:
        env.setState(starting_point)


    terminated = False
    current_score = env.score()

    #print("current_score")
    #print(current_score)

    bestScore = current_score



    while not terminated:

        env.setScore(current_score)

        action_id, input, output = strategy.choose_action(env)



        state, deltaScore, terminated = env.step(action_id)
        current_score += deltaScore

        #print("current_score")
        #print(current_score)

        current_score_v2 = env.score()

        #print("current_score_v2")
        #print(current_score_v2)

        strategy.updateInfo(action_id, env.turn)

        if current_score > bestScore:
            bestScore = current_score

        if withLogs:

            # Vous pouvez ajouter le suivi du nombre d'actions positives et supérieures ici
            # Suivre le nombre d'actions positives

            neighDeltaFitness = env.getAllDeltaFitness()

            positive_count = sum(1 for action in neighDeltaFitness if action > 0)

            # Suivre le nombre d'actions supérieures à celle choisie par le réseau
            actions_above_count = sum(1 for action in neighDeltaFitness if action > deltaScore)

            # Ajoutez cet appel pour enregistrer le score, l'action_id et sa contribution dans le fichier journal à chaque itération

            output = output.squeeze(-1).detach().numpy().tolist()



            log_file_trajectory.write(f"Turn: {env.turn}, Current Score: {current_score}, Action_id: {action_id}, Action Contribution: {deltaScore}, Positive Actions: {positive_count}, Actions Above Chosen: {actions_above_count} \n")
            input_file_trajectory.write(str(input) + "\n")
            output_file_trajectory.write(str(output) + "\n")


    if(withLogs):
        log_file_trajectory.close()
        input_file_trajectory.close()
        output_file_trajectory.close()

    return bestScore




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Optimisation de poids de réseau de neurones avec CMA-ES')

    # Ajout des arguments
    parser.add_argument('type_strategy', type=str, help='type_strategy')
    parser.add_argument('type_problem', type=str, default="NK", help='Taille de l\'instance')
    parser.add_argument('N', type=int, help='Taille de l\'instance')
    parser.add_argument('ListK', type=int, nargs='+', help='Paramètres K')
    parser.add_argument('--hiddenlayer_size', nargs='+', type=int, default=[10,5], help='Hidden layer sizes')
    parser.add_argument('--nb_restarts', type=int, default=10, help='Nombre de redémarrages')
    parser.add_argument('--nb_instances', type=int, default=10, help='Nombre d\'instances')
    parser.add_argument('--nb_jobs', type=int, default=20, help='Nombre de jobs')
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
    ListK = args.ListK


    seed = args.seed
    alpha = args.alpha
    sigma_init = args.sigma_init
    nb_restarts = args.nb_restarts
    nb_instances = args.nb_instances
    type_strategy = args.type_strategy
    max_generations = args.max_generations
    nb_jobs = args.nb_jobs
    hiddenlayer_size = args.hiddenlayer_size
    type_problem = args.type_problem

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if(nb_jobs == -1):
        nb_jobs = nb_instances*nb_restarts

    train_path_list = []

    if(args.use_trainset):

        for K in ListK:
            train_path_list.append("./benchmark/N_" + str(N) + "_K_" + str(K) + "/train/")
    else:
        for K in ListK:
            train_path_list.append("./tmp/" + type_strategy + "_seed" + str(seed) + "/")

    valid_path_list = []

    for K in ListK:
        valid_path_list.append("./benchmark/N_" + str(N) + "_K_" + str(K) + "/validation/")


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

    if(len(ListK) ==1):
        str_K = str(ListK[0])
    else:
        str_K = ""
        for idx, K in enumerate(ListK):
            if (idx == len(ListK) - 1):
                str_K += str(K)
            else:
                str_K += str(K) + ","



    # Utilisez datetime.datetime.now() pour obtenir la date actuelle
    nameResult = "test_strategy_" + type_strategy + "_" + str_ + "_N_" + str(N) + "_K_" + str_K + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(seed) + ".txt"

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
            list_strategy = [StrategyNN(N, hiddenlayer_size, remix=False ) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "strategyNNRanked_v0"):
            list_strategy = [StrategyNNRanked_v0(N, hiddenlayer_size, remix=False ) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "strategyNNRanked_v1"):
            list_strategy = [StrategyNNRanked_v1(N, hiddenlayer_size , remix=False) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "strategyNNRanked_v2"):
            list_strategy = [StrategyNNRanked_v2(N, hiddenlayer_size, remix=False ) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "strategyNNRanked_v1_zScore"):
            list_strategy = [StrategyNNRanked_v1_zScore(N, hiddenlayer_size, remix=False ) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "strategyNNRanked_v2_zScore"):
            list_strategy = [StrategyNNRanked_v2_zScore(N, hiddenlayer_size, remix=False ) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "strategyNNFitness"):
            list_strategy = [StrategyNNFitness(N, hiddenlayer_size, remix=False ) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "StrategyNNFitness_and_current"):
            list_strategy = [StrategyNNFitness_and_current(N, hiddenlayer_size, remix=False ) for idx_run in range(nb_instances * nb_restarts)]

        elif (type_strategy == "strategyNNRanked_delta_rescale"):
            list_strategy = [StrategyNNRanked_delta_rescale(N, hiddenlayer_size, remix=False ) for idx_run in range(nb_instances * nb_restarts)]



        # if(type_strategy == "strategyNNSoftmax"):
        #     list_strategy = [StrategyNNSoftmax(N, hiddenlayer_size, True, True ) for idx_run in range(nb_instances * nb_restarts)]

        # elif(type_strategy == "strategyNNnoAverageInteraction"):
        #     list_strategy = [StrategyNN(N, hiddenlayer_size, False ) for idx_run in range(nb_instances * nb_restarts)]

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
                for K in ListK:
                    Nk_generator(N, K, nb_instances, train_path_list[0])

                print("end instances generation")

            solutions = es.ask()  # Échantillonnez de nouveaux vecteurs de poids

            # Évaluez les performances de chaque solution en parallèle
            list_training_scores = []

            best_current_score = float("-inf")

            for idx, solution in enumerate(solutions):
                print("Eval individu " + str(idx))


                for idx_run in range(nb_instances * nb_restarts):
                    list_strategy[idx_run].update_weights(solution)

                all_list_scores = []
                for idx_K, K in enumerate(ListK):
                    list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)("NK",list_strategy[idx_run], N, K, train_path_list[idx_K], nb_instances, idx_run) for idx_run
                        in range(nb_instances * nb_restarts))
                    all_list_scores.extend(list_scores)


                average_training_score = np.mean(all_list_scores)

                list_training_scores.append(average_training_score)

                if average_training_score > best_current_score:

                    best_current_score = average_training_score
                    best_current_solution = np.copy(solution)


            # Mettez à jour CMA-ES avec les performances
            es.tell(solutions, -np.array(list_training_scores))

            list_strategy[0].update_weights(best_current_solution)

            # Évaluez la meilleure solution sur l'ensemble de validation

            all_list_scores = []
            for idx_K, K in enumerate(ListK):
                list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)("NK", list_strategy[0], N, K, valid_path_list[idx_K], nb_instances, idx_run) for idx_run  in range(nb_instances * nb_restarts))
                all_list_scores.extend(list_scores)
                print("Score moyen sur l'ensemble de validation pour N " + str(N) + " K " + str(K) + " : " + str(np.mean(list_scores)))

            average_validation_score = np.mean(all_list_scores)

            print("Score moyen global sur l'ensemble de validation : " + str(average_validation_score))

            f = open(pathResult + nameResult, "a")
            f.write(str(generation) + "," + str(max(list_training_scores)) + "," + str(average_validation_score) + "\n")
            f.close()

            if(average_validation_score > best_global_validation_score):
                best_global_validation_score = average_validation_score
                np.savetxt("solutions/best_solution_" + nameResult + ".csv" , best_current_solution)



    elif(type_strategy == "tabu" or type_strategy == "iteratedHillClimber" or type_strategy == "oneLambdaDeterministic"):


        best_current_score = float("-inf")

        if (args.use_trainset == False):
            print("start instances generation")
            # Générer de nouvelles instances de formation à chaque itération
            for K in ListK:
                Nk_generator(N, K, nb_instances, train_path_list[0])
            print("end instances generation")

        for paramValue in range(1,N):

            if(type_strategy == "tabu"):
                list_strategy = [Tabu(N, paramValue) for idx_run in range(nb_instances * nb_restarts)]
            elif(type_strategy == "iteratedHillClimber"):
                list_strategy = [IteratedHillClimber(N, paramValue) for idx_run in range(nb_instances * nb_restarts)]
            elif(type_strategy == "oneLambdaDeterministic"):
                list_strategy = [OneLambdaDeterministic(N, paramValue) for idx_run in range(nb_instances * nb_restarts)]


            # Évaluez les performances de chaque solution en parallèle
            list_training_scores = []

            all_list_scores = []
            for idx_K, K in enumerate(ListK):
                list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[idx_run], N, K, train_path_list[idx_K], nb_instances, idx_run) for idx_run
                    in range(nb_instances * nb_restarts))
                all_list_scores.extend(list_scores)

            average_training_score = np.mean(all_list_scores)
            print("paramValue : " + str(paramValue) + " - average_training_score : " + str(average_training_score))

            if average_training_score > best_current_score:
                best_current_score = average_training_score
                best_paramValue = paramValue


        if(type_strategy == "tabu"):
            list_strategy = [Tabu(N, best_paramValue) for idx_run in range(nb_instances * nb_restarts)]
        elif(type_strategy == "iteratedHillClimber"):
            list_strategy = [IteratedHillClimber(N, best_paramValue) for idx_run in range(nb_instances * nb_restarts)]
        elif(type_strategy == "oneLambdaDeterministic"):
            list_strategy = [OneLambdaDeterministic(N, best_paramValue) for idx_run in range(nb_instances * nb_restarts)]

        # Évaluez la meilleure solution sur l'ensemble de validation
        all_list_scores = []
        for idx_K, K in enumerate(ListK):
            list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[0], N, K, valid_path_list[idx_K], nb_instances, idx_run) for idx_run  in range(nb_instances * nb_restarts))
            print("Score moyen sur l'ensemble de validation pour N " + str(N) + " K " + str(K) + " : " + str(
                np.mean(list_scores)))
            all_list_scores.extend(list_scores)

        average_validation_score = np.mean(all_list_scores)

        print("Score global moyen sur l'ensemble de validation : " + str(average_validation_score))
        print("best paramValue : " + str(best_paramValue))

        f = open(pathResult + nameResult , "a")
        f.write(str(best_paramValue) + "," + str(average_training_score) + "," + str(average_validation_score) + "\n")
        f.close()


    else:


        print("Évaluation de la stratégie " + type_strategy)

        if(type_strategy == "hillClimber"):
            list_strategy = [HillClimber(N) for idx_run in range(nb_instances * nb_restarts)]

        if(type_strategy == "hillClimberJump"):
            list_strategy = [HillClimberJump(N) for idx_run in range(nb_instances * nb_restarts)]

        if(type_strategy == "hillClimberFirstImprovementJump"):
            list_strategy = [HillClimberFirstImprovementJump(N) for idx_run in range(nb_instances * nb_restarts)]

        if(type_strategy == "emergingDeterministicStrategy"):
            list_strategy = [EmergingDeterministicStrategyK8(N) for idx_run in range(nb_instances * nb_restarts)]




        # elif (type_strategy == "iteratedHillClimber"):
        #     list_strategy = [IteratedHillClimber(N) for idx_run in range(nb_instances * nb_restarts)]

        all_list_scores = []
        for idx_K, K in enumerate(ListK):
            list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(type_problem, list_strategy[idx_run], N, K, valid_path_list[idx_K], nb_instances, idx_run, alpha=alpha) for idx_run in
            range(nb_instances * nb_restarts))
            print("Score moyen sur l'ensemble de validation pour N " + str(N) + " K " + str(K) + " : " + str(
                np.mean(list_scores)))

            all_list_scores.extend(list_scores)


        average_score_baseline = np.mean(all_list_scores)

        print("Score moyen de la stratégie " + type_strategy + " sur l'ensemble de validation :")
        print(average_score_baseline)
        f = open(pathResult + nameResult, "a")
        f.write(str(0) + ",," + str(average_score_baseline) + "\n")
        f.close()

