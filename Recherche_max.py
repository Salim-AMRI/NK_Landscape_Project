import numpy as np
from joblib import Parallel, delayed
import argparse

from Main import get_Score_trajectory
from strategies.HillClimber import HillClimber
from strategies.StrategyNN import StrategyNN
from strategies.strategyNNLastMoveIndicatorTabu import StrategyNNLastMoveIndicatorTabu
from strategies.Tabu import Tabu
import os
from scipy import stats
parser = argparse.ArgumentParser(description='Optimisation de poids de réseau de neurones avec CMA-ES')

parser.add_argument('--nb_restarts', type=int, default=5, help='Nombre de redémarrages')
parser.add_argument('--nb_instances', type=int, default=10, help='Nombre d\'instances')
parser.add_argument('--nb_jobs', type=int, default=-1, help='Nombre de jobs')

args = parser.parse_args()

# Paramètres initiaux en fonction des argumentssolution = np.loadtxt(solution_file)

nb_restarts = args.nb_restarts
nb_instances = args.nb_instances
nb_jobs = args.nb_jobs

# Chemin du répertoire contenant les fichiers à consulter
directory = 'results_09102023/'

# Initialisation de la valeur maximale et du nom du fichier correspondant
max_value = None
file_with_max_value = None

for N in [32, 64, 128]:
    for K in [1, 2, 4, 8]:

        list_list_strategy = []

        for type_strategy in ["hillClimber", "strategyNN", "Tabu", "strategyNNLastMoveIndicatorTabu"]:

            if("NN" in type_strategy):
                # Parcours des fichiers dans le répertoire
                for filename in os.listdir(directory):

                    if filename.endswith('.txt') and 'test_strategy_' + type_strategy + '_10,5_N_' + str(N) + '_K_' + str(K) in filename:
                        file_path = os.path.join(directory, filename)

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

                # Affiche le nom du fichier avec la valeur maximale et la valeur maximale
                if max_value is not None:
                    print(f"Le fichier '{file_with_max_value}' contient la valeur maximale : {max_value}")
                else:
                    print("Aucune valeur maximale trouvée dans les fichiers.")


                name = 'best_solution_' + file_with_max_value + '.csv'


                path = "solutions_09102023/"

                splitName = name.split("_")

                solution = np.loadtxt(path + name)

                print(splitName)
                hiddenLayer_str =  splitName[5]

                hiddenlayer_size = []
                split_hiddenLayer_str = hiddenLayer_str.split(",")

                print(split_hiddenLayer_str)

                for layer in split_hiddenLayer_str:
                    hiddenlayer_size.append(int(layer))

                print("hiddenlayer_size")
                print(hiddenlayer_size)

                # Chargez les paramètres de la solution optimale à partir du fichier CSV

            if (nb_jobs == -1):
                nb_jobs = nb_instances * nb_restarts

            if (type_strategy == "strategyNN"):
                list_strategy = [StrategyNN(N, hiddenlayer_size) for idx_run in range(nb_instances * nb_restarts)]

            elif (type_strategy == "strategyNNLastMoveIndicatorTabu"):
                list_strategy = [StrategyNNLastMoveIndicatorTabu(N, hiddenlayer_size) for idx_run in
                                 range(nb_instances * nb_restarts)]

            elif(type_strategy == "hillClimber"):
                list_strategy = [HillClimber(N) for idx_run in range(nb_instances * nb_restarts)]

            elif type_strategy == "tabu":
                # Initialize Tabu_Time to None
                Tabu_Time = None
                max_value = None

                # Define the directory where result files are located
                result_directory = 'results_09102023/'

                for filename in os.listdir(result_directory):
                    if filename.endswith('.txt') and 'test_strategy_' + type_strategy + '_10,5N_' + str(N) + '_K_' + str(K) in filename:
                        file_path = os.path.join(result_directory, filename)

                        with open(file_path, 'r') as file:
                            lines = file.readlines()

                            for line in lines:
                                elements = line.strip().split(',')
                                if len(elements) >= 3:
                                    try:
                                        value = float(elements[2])

                                        if Tabu_Time is None or value > max_value:
                                         Tabu_Time = int(elements[0].split(',')[0])
                                         max_value = value

                                    except ValueError:
                                        pass

                if Tabu_Time is not None:
                    print(f"Tabu_Time is set to: {Tabu_Time}")
                else:
                    print("No Tabu_Time value found in the files.")

                list_strategy = [Tabu(N, Tabu_Time) for idx_run in range(nb_instances * nb_restarts)]

            if("NN" in type_strategy):
                    for idx_run in range(nb_instances * nb_restarts):
                        list_strategy[idx_run].update_weights(solution)

            # Chemin vers le répertoire contenant les instances de test
            path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/test/"

            # Utilisez la bibliothèque joblib pour paralléliser l'évaluation sur les instances
            list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[idx_run], N, K, path, nb_instances, idx_run, alpha=None, withLogs=True) for idx_run in range(nb_instances * nb_restarts))


            list_list_strategy.append(list_scores)
            print(type_strategy + " : " + str(np.mean(list_scores)/ + N))


        print(stats.ttest_ind(list_list_strategy[0], list_list_strategy[1]))
        print(stats.ttest_ind(list_list_strategy[2], list_list_strategy[3]))
'''
# Créez un tableau LaTeX avec les valeurs
latex_table = "\\begin{tabular}{|c|c|c|}\n"
latex_table += "\\hline\n"
latex_table += "Instance & Type de Stratégie & Moyenne des Scores \\\\\n"
latex_table += "\\hline\n"

for N in [32, 64, 128]:
    for K in [1, 2, 4, 8]:
        for i, type_strategy in enumerate(["hillClimber", "strategyNN", "strategyNNLastMoveIndicatorTabu", "tabu"]):
            instance_name = f"N_{N}_K_{K}"
            latex_table += f"{instance_name} & {type_strategy} & {np.mean(list_list_strategy[i]):.2f} \\\\\n"
            latex_table += "\\hline\n"

latex_table += "\\end{tabular}"

# Définissez le chemin du fichier LaTeX de sortie
output_tex_file = "tableau_resultats.tex"

# Écrivez le contenu LaTeX dans le fichier
with open(output_tex_file, "w") as tex_file:
    tex_file.write(latex_table)

print(f"Tableau des résultats LaTeX écrit dans {output_tex_file}")
'''