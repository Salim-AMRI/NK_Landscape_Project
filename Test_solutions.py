import numpy as np
import torch
from joblib import Parallel, delayed
import argparse

from Main import get_Score_trajectory
from strategies.HillClimber import HillClimber
from strategies.StrategyNN import StrategyNN
from strategies.strategyNNLastMoveIndicatorTabu import StrategyNNLastMoveIndicatorTabu
from strategies.Tabu import Tabu

parser = argparse.ArgumentParser(description='Optimisation de poids de réseau de neurones avec CMA-ES')



# parser.add_argument('type_strategy', type=str, help='type_strategy')
# parser.add_argument('--hiddenlayer_size', nargs='+', type=int, default=[10,5], help='Hidden layer sizes')
# parser.add_argument('N', type=int, help='Taille de l\'inst
# ance')
# parser.add_argument('K', type=int, help='Paramètre K')

parser.add_argument('name', type=str, help='nameArgument')

parser.add_argument('--nb_restarts', type=int, default=5, help='Nombre de redémarrages')
parser.add_argument('--nb_instances', type=int, default=10, help='Nombre d\'instances')
parser.add_argument('--nb_jobs', type=int, default=-1, help='Nombre de jobs')




args = parser.parse_args()

# Paramètres initiaux en fonction des argumentssolution = np.loadtxt(solution_file)

name = args.name

nb_restarts = args.nb_restarts
nb_instances = args.nb_instances
nb_jobs = args.nb_jobs

path = "solutions/"


if (nb_jobs == -1):
    nb_jobs = nb_instances * nb_restarts

# Chemin du fichier CSV contenant la solution optimale
#solution = np.loadtxt("solutions_02102023/best_solution_test_strategy_InvariantNN_32_K_1_2023-09-29_18-17-30_0.txt.csv")

splitName = name.split("_")

solution = np.loadtxt(path + name)


print(splitName)

type_strategy = splitName[4]
hiddenLayer_str =  splitName[5]
N =  int(splitName[7])
K =  int(splitName[9])


hiddenlayer_size = []
split_hiddenLayer_str = hiddenLayer_str.split(",")

print(split_hiddenLayer_str)

for layer in split_hiddenLayer_str:
    hiddenlayer_size.append(int(layer))

print("hiddenlayer_size")
print(hiddenlayer_size)

# Chargez les paramètres de la solution optimale à partir du fichier CSV
print("solution")
print(solution)

if (type_strategy == "strategyNN"):
    list_strategy = [StrategyNN(N, hiddenlayer_size) for idx_run in range(nb_instances * nb_restarts)]
elif (type_strategy == "strategyNNLastMoveIndicatorTabu"):
    list_strategy = [StrategyNNLastMoveIndicatorTabu(N, hiddenlayer_size) for idx_run in
                     range(nb_instances * nb_restarts)]
elif(type_strategy == "hillClimber"):
    list_strategy = [HillClimber(N) for idx_run in range(nb_instances * nb_restarts)]
elif(type_strategy == "tabu"):
    list_strategy = [Tabu(N) for idx_run in range(nb_instances * nb_restarts)]

if("NN" in type_strategy):
    for idx_run in range(nb_instances * nb_restarts):
        list_strategy[idx_run].update_weights(solution)



# Chemin vers le répertoire contenant les instances de test
path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/validation/"


# Utilisez la bibliothèque joblib pour paralléliser l'évaluation sur les instances
list_scores = Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(list_strategy[idx_run], N, K, path, nb_instances, idx_run, alpha=None, withLogs=True) for idx_run in range(nb_instances * nb_restarts))

print(list_scores)