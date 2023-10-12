import numpy as np
from joblib import Parallel, delayed
import argparse
from Main import get_Score_trajectory
from strategies.HillClimberJump import HillClimberJump
from strategies.OneLambdaDeterministic import OneLambdaDeterministic
from strategies.StrategyNN import StrategyNN
from strategies.HillClimberFirstImprovementJump import HillClimberFirstImprovementJump
import os
from scipy import stats
from scipy.stats import shapiro

from strategies.Tabu import Tabu

parser = argparse.ArgumentParser(description='Optimisation de poids de réseau de neurones avec CMA-ES')

parser.add_argument('--nb_restarts', type=int, default=1, help='Nombre de redémarrages')
parser.add_argument('--nb_instances', type=int, default=100, help='Nombre d\'instances')
parser.add_argument('--nb_jobs', type=int, default=20, help='Nombre de jobs')

args = parser.parse_args()

# Paramètres initiaux en fonction des argumentssolution = np.loadtxt(solution_file)

nb_restarts = args.nb_restarts
nb_instances = args.nb_instances
nb_jobs = args.nb_jobs

# Chemin du répertoire contenant les fichiers à consulter
#directory = 'results_09102023/'


result_directory = 'results'
solution_directory = "solutions/"

# Initialisation de la valeur maximale et du nom du fichier correspondant
max_value = None
file_with_max_value = None

results_dict = {}


f = open("images/results.txt", "w")
f.write("Results test")
f.close()



list_strategy_name = ["hillClimberJump", "hillClimberFirstImprovementJump", "oneLambdaDeterministic",
                              "strategyNN"]
#
# for N in [32, 64, 128]:
#     for K in [1, 2, 4, 8, 'all']:

for N in [32]:
    for K in ['all']:

        if(K == 'all'):
            list_starting_points = [np.random.randint(2, size=N) for i in range(nb_instances*nb_restarts)]
            list_K = [1,2,4,8]
            str_K = "1,2,4,8"
        else:
            list_starting_points = [np.random.randint(2, size=N) for i in range(nb_instances * nb_restarts*4)]
            list_K = [K]
            str_K = str(K)

        results_dict[f'N_{N}_K_' + str_K] = {}

        print("N : " + str(N) + " K : " + str_K)

        list_all_strategy = []

        for type_strategy in list_strategy_name:



            if("NN" in type_strategy):
                # Parcours des fichiers dans le répertoire
                for filename in os.listdir(result_directory):

                    if filename.endswith('.txt') and 'test_strategy_' + type_strategy + '_10,5_N_' + str(N) + '_K_' + str_K in filename:
                        file_path = os.path.join(result_directory, filename)

                        # Ouvre le fichier en mode lecture
                        with open(file_path, 'r') as file:
                            lines = file.readlines()

                            for line in lines:
                                # Sépare la ligne en éléments en utilisant la virgule comme séparateur
                                elements = line.strip().split(',')

                                # Vérifie s'il y a au moins 3 éléments (après la deuxième virgule)
                                if len(elements) >= 3:
                                    try:
                                        # Convertit le troisième élément en un nombre flottant
                                        value = float(elements[2])

                                        # Si max_value est None ou si la valeur est supérieure à max_value, met à jour max_value
                                        if max_value is None or value > max_value:
                                           max_value = value
                                           file_with_max_value = filename

                                    except ValueError:
                                        # En cas d'erreur de conversion, ignore cette ligne
                                        pass

                '''
                # Affiche le nom du fichier avec la valeur maximale et la valeur maximale
                if max_value is not None:
                    print(f"Le fichier '{file_with_max_value}' contient la valeur maximale : {max_value}")
                else:
                    print("Aucune valeur maximale trouvée dans les fichiers.")
                '''

                name = 'best_solution_' + file_with_max_value + '.csv'


                #path = "solutions_09102023/"


                splitName = name.split("_")

                solution = np.loadtxt(solution_directory + name)


                hiddenLayer_str =  splitName[5]

                hiddenlayer_size = []
                split_hiddenLayer_str = hiddenLayer_str.split(",")


                for layer in split_hiddenLayer_str:
                    hiddenlayer_size.append(int(layer))

                # Chargez les paramètres de la solution optimale à partir du fichier CSV

            if (nb_jobs == -1):
                nb_jobs = nb_instances * nb_restarts

            if (type_strategy == "strategyNN"):
                list_strategy = [StrategyNN(N, hiddenlayer_size) for idx_run in range(nb_instances * nb_restarts)]

            elif(type_strategy == "hillClimberJump"):
                list_strategy = [HillClimberJump(N) for idx_run in range(nb_instances * nb_restarts)]

            elif (type_strategy == "hillClimberFirstImprovementJump"):
                list_strategy = [HillClimberFirstImprovementJump(N) for idx_run in range(nb_instances * nb_restarts)]

            elif type_strategy == "oneLambdaDeterministic":

                lambda_ = None
                max_value = None

                for filename in os.listdir(result_directory):
                    if filename.endswith('.txt') and 'test_strategy_' + type_strategy + '_10,5_N_' + str(
                            N) + '_K_' + str_K in filename:


                        file_path = os.path.join(result_directory, filename)

                        with open(file_path, 'r') as file:

                            lines = file.readlines()
                            for line in lines:
                                elements = line.strip().split(',')
                                if len(elements) >= 3:
                                    try:
                                        value = float(elements[2])
                                        if max_value is None or value > max_value:
                                            lambda_ = int(elements[0].split(',')[0])
                                            max_value = value  # Mettez à jour max_value ici
                                    except ValueError:
                                        pass

                if lambda_ is not None:
                    print(f"Lambda is set to: {lambda_}")
                else:
                    print("No Lambda value found in the files.")

                list_strategy = [OneLambdaDeterministic(N, int(lambda_)) for idx_run in range(nb_instances * nb_restarts)]


            if("NN" in type_strategy):
                    for idx_run in range(nb_instances * nb_restarts):
                        list_strategy[idx_run].update_weights(solution)


            list_all_scores = []
            for K in list_K:
                # Chemin vers le répertoire contenant les instances de test
                path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/test/"

                # Utilisez la bibliothèque joblib pour paralléliser l'évaluation sur les instances
                list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[idx_run], N, K, path, nb_instances, idx_run, alpha=None, withLogs=True, starting_point = list_starting_points[idx_run]) for idx_run in range(nb_instances * nb_restarts))

                print("Score moyen sur l'ensemble de test pour N " + str(N) + " K " + str(K) + " : " + str(np.mean(list_scores)/N))

                list_all_scores.extend(list_scores)

                #print(list_scores)

            list_all_strategy.append(list_all_scores)
            print(type_strategy + " : " + str(np.mean(list_all_scores)/ N))


        # Enregistrer les résultats dans un fichier texte
        with open("images/results.txt", "a") as result_file:

            result_file.write("N " + str(N) + " K " + str(K) + ":\n")
            for idx, strategy in enumerate(list_strategy_name):
                mean_score = np.mean(list_all_strategy[idx])/ N
                result_file.write(f"  {strategy}: {mean_score}\n")

        '''
        print("shapiro")
        print(shapiro(list_list_strategy[0]))
        print(shapiro(list_list_strategy[1]))
        print(shapiro(list_list_strategy[2]))
        print(shapiro(list_list_strategy[3]))

        print(stats.ttest_ind(list_list_strategy[0], list_list_strategy[1], list_list_strategy[2], list_list_strategy[3]))
        #print(stats.ttest_ind(list_list_strategy[2], list_list_strategy[3]))
        '''