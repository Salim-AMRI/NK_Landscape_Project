# Importation de bibliotheque
import numpy as np
import pandas as pd
import copy
import argparse
import datetime
import torch
import cma
import os
from joblib import Parallel, delayed
#from tqdm import tqdm
import random

# Importation de modules personnalisés
from neural_net import Net, InvariantNNet
from env_nk_landscape import EnvNKlandscape
from Instances_Generator import Nk_generator
import torch.nn.functional as F


# Création du parser d'arguments
parser = argparse.ArgumentParser(description='Optimisation de poids de réseau de neurones avec CMA-ES')

# Ajout des arguments
parser.add_argument('type_strategy', type=str, help='type_strategy')
parser.add_argument('N', type=int, help='Taille de l\'instance')
parser.add_argument('K', type=int, help='Paramètre K')
parser.add_argument('--nb_restarts', type=int, default=5, help='Nombre de redémarrages')
parser.add_argument('--nb_instances', type=int, default=10, help='Nombre d\'instances')
parser.add_argument('--nb_jobs', type=int, default=-1, help='Nombre de jobs')
parser.add_argument('--sigma_init', type=float, default=0.5, help='Ecart-type initial')
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

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if(nb_jobs == -1):
    nb_jobs = nb_instances*nb_restarts

if(args.use_trainset):
    train_path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/train/"
else:
    train_path = "./tmp/seed" + str(seed) + "/"

valid_path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/validation/"
pathResult = "results/"


if not os.path.exists("solutions"):
    os.makedirs("solutions")
if not os.path.exists("tmp"):
    os.makedirs("tmp")


# Utilisez datetime.datetime.now() pour obtenir la date actuelle
nameResult = "test_strategy_" + type_strategy + "_" + str(N) + "_K_" + str(K) + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(seed) + ".txt"
f = open(os.path.join(pathResult, nameResult), "w")
f.write("generation,avg_training_score,avg_validation_score\n")
f.close()

hillClimber_results = []
IteratedhillClimber_results = []
tabou_results = []

def get_Score_trajectory(type_strategy, N, K, network, path, nb_intances, idx_run, alpha=None):


    i = idx_run%nb_intances
    name_instance = path + "nk_" + str(N) + "_" + str(K) + "_" + str(i) + ".txt"

    #print("launch instance " + name_instance)
    env = EnvNKlandscape(name_instance)
    env.reset()
    terminated = False
    current_score = env.score()
    bestScore = current_score

    while not terminated:
        # Extraction des observations à partir de l'état
        neigh = []

        if("rawInput" in type_strategy):
            for i in range(N):
                neigh.append(env.getDeltaFitness(i) + current_score)
        else:
            for i in range(N):
                neigh.append(env.getDeltaFitness(i))



        if type_strategy == "NN":
            # Stratégie basée sur le réseau neuronal (NN)
            non_ordered_neigh = copy.deepcopy(neigh)
            neigh.sort(reverse=True)
            neigh = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0)
            out = network(neigh.unsqueeze(0)).squeeze(0)
            action = out.argmax().item()
            action_id = non_ordered_neigh.index(neigh[0][action])

        if type_strategy == "NNsoftmax":
            # Stratégie basée sur le réseau neuronal (NN)
            non_ordered_neigh = copy.deepcopy(neigh)
            neigh.sort(reverse=True)
            neigh = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0)
            out = network(neigh.unsqueeze(0)).squeeze(0)
            test = F.gumbel_softmax(out, tau=1, hard=True)
            action = test.argmax().item()
            action_id = non_ordered_neigh.index(neigh[0][action])
            
        elif type_strategy == "NN_withTabu":
            # Stratégie basée sur le réseau neuronal (NN with Tabu)
            non_ordered_neigh = copy.deepcopy(neigh)
            tabuList = env.getTabuList()
            df = pd.DataFrame()
            df["fitness"] = neigh
            df["tabulist"] = tabuList
            df = df.sort_values(by="fitness")
            sorted_neighbors = df["fitness"].values
            sorted_tabuList = df["tabulist"].values
            stacked_input = np.hstack((sorted_neighbors, sorted_tabuList))
            stacked_input = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
            out = network(stacked_input.unsqueeze(0)).squeeze(0)
            action = out.argmax().item()
            action_id = non_ordered_neigh.index(sorted_neighbors[action])

        elif type_strategy == "NNsoftmax_withTabu":
            # Stratégie basée sur le réseau neuronal (NN with Tabu)
            non_ordered_neigh = copy.deepcopy(neigh)
            tabuList = env.getTabuList()
            df = pd.DataFrame()
            df["fitness"] = neigh
            df["tabulist"] = tabuList
            df = df.sort_values(by="fitness")
            sorted_neighbors = df["fitness"].values
            sorted_tabuList = df["tabulist"].values
            stacked_input = np.hstack((sorted_neighbors, sorted_tabuList))
            stacked_input = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
            out = network(stacked_input.unsqueeze(0)).squeeze(0)
            test = F.gumbel_softmax(out, tau=1, hard=True)
            action = test.argmax().item()
            action_id = non_ordered_neigh.index(sorted_neighbors[action])

        elif type_strategy == "NN_unsorted":

            neigh = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0)
            out = network(neigh.unsqueeze(0)).squeeze(0)
            action_id = out.argmax().item()
                    
        elif type_strategy == "NN_withTabu_unsorted":

            neigh = torch.tensor(neigh, dtype=torch.float32)
            tabuList = env.getTabuList()
            stacked_input = np.hstack((np.array(neigh), np.array(tabuList)))
            
            stacked_input = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
            

            out = network(stacked_input)
            

            action_id = out.argmax().item()


        elif type_strategy == "InvariantNN":

            stacked_input_th = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            out = network(stacked_input_th).squeeze(0)

            action_id = out.argmax().item()

        elif type_strategy == "InvariantNN_softmax":


            stacked_input_th = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            out = network(stacked_input_th).squeeze(0)
            test = F.gumbel_softmax(out, tau=1, hard=True)
            action_id = test.argmax().item()

        elif type_strategy == "InvariantNN_withTabu":

            tabuList = env.getTabuList()
            stacked_input = np.vstack((neigh, tabuList))
            stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
            stacked_input_th = torch.transpose(stacked_input_th, 1, 2)

            out = network(stacked_input_th).squeeze(0)

            action_id = out.argmax().item()

        elif type_strategy == "InvariantNN_withTabu_softmax":

            tabuList = env.getTabuList()
            stacked_input = np.vstack((neigh, tabuList))
            stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)

            stacked_input_th = torch.transpose(stacked_input_th, 1, 2)

            out = network(stacked_input_th).squeeze(2).squeeze(0)

            test = F.gumbel_softmax(out, tau=1, hard=True)

            action_id = test.argmax().item()


        elif type_strategy == "InvariantNN_rawInput":

            stacked_input_th = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            out = network(stacked_input_th).squeeze(0)

            action_id = out.argmax().item()

        elif type_strategy == "InvariantNN_withTabu_rawInput":

            tabuList = env.getNormalizedTabuTurn()
            stacked_input = np.vstack((neigh, tabuList))
            stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
            stacked_input_th = torch.transpose(stacked_input_th, 1, 2)

            out = network(stacked_input_th).squeeze(0)

            action_id = out.argmax().item()
            

        elif type_strategy == "InvariantNN_rawInput_generalInfo":

            currentScore_np = np.ones(N) * current_score
            bestScore_np = np.ones(N) * bestScore
            
            stacked_input = np.vstack((neigh, currentScore_np, bestScore_np))
            
            #print("stacked_input")
            #print(stacked_input)
            #print(stacked_input.shape)
            
            stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
            stacked_input_th = torch.transpose(stacked_input_th, 1, 2)

            out = network(stacked_input_th).squeeze(0)

            action_id = out.argmax().item()
            
            
        elif type_strategy == "InvariantNN_withTabu_rawInput_generalInfo":

            #currentTurn_np = np.ones(N) * env.getTurn()
            
            currentScore_np = np.ones(N) * current_score
            bestScore_np = np.ones(N) * bestScore

            
            tabuList = env.getNormalizedTabuTurn()
            #tabuList =  env.getNormalizedTabuTurn()
            stacked_input = np.vstack((neigh, tabuList, currentScore_np, bestScore_np))
            
            #print("stacked_input")
            #print(stacked_input)
            #print(stacked_input.shape)
            
            stacked_input_th = torch.tensor(stacked_input, dtype=torch.float32).unsqueeze(0)
            stacked_input_th = torch.transpose(stacked_input_th, 1, 2)

            out = network(stacked_input_th).squeeze(0)

            action_id = out.argmax().item()
            

        elif type_strategy == "hillClimber":
            # Stratégie HillClimber
            action_id = int(np.argmax(np.array(neigh)))
            # Collecter les résultats ici
            #hillClimber_results.append((generation, current_score, action_id))

        elif type_strategy == "hillClimberSoftmax":
            # Stratégie HillClimber
            
            neigh = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0)
            test = F.gumbel_softmax(neigh, tau=1, hard=True)
            action_id = test.argmax().item()
            
            #action_id = int(np.argmax(np.array(neigh)))
            # Collecter les résultats ici
            #hillClimber_results.append((generation, current_score, action_id))
            
            
        elif type_strategy == "IteratedhillClimber":
            # Stratégie HillClimber itératif
            if max(neigh) > 0:
                action_id = int(np.argmax(np.array(neigh)))
                # Collecter les résultats ici
                #IteratedhillClimber_results.append((generation, current_score, action_id))
            else:
                action_id = -1
                # Collecter les résultats ici
                #IteratedhillClimber_results.append((generation, current_score, action_id))

        elif type_strategy == "tabu":
            # Stratégie Tabu
            tabuList = env.getTabuList()
            low_values = np.ones(tabuList.shape) * float("-inf")
            filter_neigh = np.where(tabuList < 1, neigh, low_values )
            action_id = int(np.argmax(np.array(filter_neigh)))

            ### Critere aspiration
            # best_delta = -99999
            # action_id = -1
            # for i in range(N):
            #     delta = neigh[i]
            #     if(tabuList[i] == 0 or delta + current_score > bestScore):
            #         if(delta > best_delta):
            #             best_delta = delta
            #             action_id = i

        if action_id >= 0:
            state, deltaScore, terminated = env.step(action_id)
            current_score += deltaScore

        # action de réaliser une perturbation
        elif action_id == -1:
            env.perturbation(alpha)
            current_score = env.score()

        # Collecter les résultats ici
        #tabou_results.append((generation, current_score, action_id))

        if current_score > bestScore:
            bestScore = current_score

    return bestScore

# Fonction pour calculer le score moyen d'une stratégie
def get_average_score_strategy(type_strategy, N, K, weights, network, path, nb_instances, nb_restarts, nb_jobs, alpha=None):
    # Cloner le réseau neuronal pour éviter de modifier les poids originaux

    if "NN" in type_strategy:
        # Charger les poids dans le réseau neuronal cloné
        for i, param in enumerate(network.parameters()):
            param.data.copy_(torch.from_numpy(weights[i]).type(torch.FloatTensor))

    average_score = 0

    list_scores =  Parallel(n_jobs=nb_jobs)(delayed(get_Score_trajectory)(type_strategy, N, K, network, path, nb_instances, idx_run, alpha=alpha) for idx_run in range(nb_instances*nb_restarts))

    for score in list_scores:
        average_score += score

    return average_score / (nb_instances * nb_restarts)

# Fonction d'évaluation pour CMA-ES
def evaluate_weights_NN(type_strategy, N, K, solution, network, path, nb_instances, nb_restarts, nb_jobs, alpha=None):
    # Assurez-vous que la taille du vecteur solution correspond à num_params
    assert len(solution) == num_params
    params_net = list(network.parameters())
    # Remodelez le vecteur de poids en tant que paramètres du réseau de neurones
    start = 0
    for param in params_net:
        end = start + param.numel()
        new_param = torch.from_numpy(solution[start:end]).type(torch.FloatTensor)
        new_param = new_param.view(param.size())  # Assurez-vous que les dimensions correspondent
        param.data.copy_(new_param)
        start = end

    # Extrayez les poids du réseau de neurones sous forme de numpy.ndarray
    weights = [param.data.numpy() for param in params_net]

    # Évaluez le réseau de neurones avec les nouveaux poids
    average_score = get_average_score_strategy(type_strategy, N, K, weights, network, path, nb_instances, nb_restarts,nb_jobs, alpha)
    return average_score


## Add save result
if "NN" in type_strategy:
    # Création de l'architecture du réseau de neurones
    if("InvariantNN" in type_strategy):

        if("generalInfo" in type_strategy ):
        
            if("Tabu" in type_strategy):
                layers_size = [4, 10, 5, 1]
            else:
                layers_size = [3, 10, 5, 1]
                
        else:
            if ("Tabu" in type_strategy):
                layers_size = [2, 10, 5, 1]
            else:
                layers_size = [1, 10, 5, 1]
            
        
        
        
            
        #if ("Tabu" in type_strategy):
            #layers_size = [2, 20, 20 , 10, 5, 1]
        #else:
            #layers_size = [1, 20, 20 , 10, 5, 1]
            
    else:
        if("Tabu" in type_strategy ):
            layers_size = [2 * N, 2 * N, N]
        else:
            layers_size = [N, 2 * N, N]

    # Initialisation du réseau de neurones
    # Initialisation du réseau de neurones
    if (  "InvariantNN" in type_strategy):
        nnet = InvariantNNet(N, True, layers_size)
    else:
        nnet = Net(layers_size)


    params_net = list(nnet.parameters())

    # Paramètres CMA-ES
    #sigma_init = 0.2  # Écart-type initial
    num_params = sum(p.numel() for p in params_net)  # Nombre total de paramètres dans le réseau de neurones

    # Initialisation de CMA-ES avec un vecteur de poids initial aléatoire
    initial_solution = np.random.randn(num_params)
    es = cma.CMAEvolutionStrategy(initial_solution, sigma_init, {'seed':seed})

    print("Taille de la population dans CMA-ES :", es.popsize)

    # Création d'une liste de réseaux de neurones pour chaque membre de la population CMA-ES
    list_nnet = [Net(layers_size) for i in range(es.popsize)]

    # Créez une barre de progression avec tqdm
    #pbar = tqdm(total=max_generations)

    # Initialisez la meilleure récompense à un score initial bas (ou négatif)

    best_global_validation_score = float("-inf")

    for generation in range(max_generations):

        print("start instances generation")
        if(args.use_trainset == False):
            # Générer de nouvelles instances de formation à chaque itération
            Nk_generator(N, K, nb_instances, train_path)
        print("end instances generation")

        solutions = es.ask()  # Échantillonnez de nouveaux vecteurs de poids

        # Évaluez les performances de chaque solution en parallèle
        training_scores = []
        for idx, solution in enumerate(solutions):
            print("Eval individu " + str(idx))
            training_scores.append(evaluate_weights_NN(type_strategy, N, K, solution, list_nnet[idx], train_path, nb_instances, nb_restarts, nb_jobs, alpha))
        best_current_score = float("-inf")

        # Mettez à jour la meilleure solution trouvée par CMA-ES
        for idx, score in enumerate(training_scores):
            if score > best_current_score:
                best_current_score = score
                best_current_solution = solutions[idx]

        # Mettez à jour CMA-ES avec les performances
        es.tell(solutions, -np.array(training_scores))

        # Évaluez la meilleure solution sur l'ensemble de validation
        validation_score = evaluate_weights_NN(type_strategy, N, K, best_current_solution, list_nnet[0], valid_path, nb_instances, nb_restarts, nb_jobs, alpha)
        print("Score moyen sur l'ensemble de validation : " + str(validation_score))

        f = open(pathResult + nameResult, "a")
        f.write(str(generation) + "," + str(max(training_scores)) + "," + str(validation_score) + "\n")
        f.close()

        if(validation_score > best_global_validation_score):
            best_global_validation_score = validation_score
            np.savetxt("solutions/best_solution_" + nameResult + ".csv" , best_current_solution)


        # Mettez à jour la barre de progression
        #pbar.set_postfix(avg_training_score=max(training_scores), avg_validation_score=validation_score)
        #pbar.update(1)  # Incrémentation de la barre de progression

else:
    print("Évaluation de la stratégie " + type_strategy)
    average_score_baseline = get_average_score_strategy(type_strategy, N, K, None, None, valid_path, nb_instances, nb_restarts, nb_jobs, alpha)
    print("Score moyen de la stratégie " + type_strategy + " sur l'ensemble de validation :")
    print(average_score_baseline)
    f = open(pathResult + nameResult, "a")
    f.write(str(0) + ",," + str(average_score_baseline) + "\n")
    f.close()

