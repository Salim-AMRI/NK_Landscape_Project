import torch
import numpy as np
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import cma
import math  # Importez le module math pour le plancher (math.floor)

# Importation de modules personnalisés
from neural_net import Net
from env_nk_landscape import EnvNKlandscape

# Fonction pour calculer le score moyen d'une stratégie
def get_average_score_strategy(type_strategy, N, K, weights, network, path, nb_instances, nb_restarts, alpha=None):
    # Cloner le réseau neuronal pour éviter de modifier les poids originaux

    if type_strategy == "NN":
        # Charger les poids dans le réseau neuronal cloné
        for i, param in enumerate(network.parameters()):
            param.data.copy_(torch.from_numpy(weights[i]).type(torch.FloatTensor))

    average_score = 0

    for i in range(nb_instances):
        name_instance = path + "nk_" + str(N) + "_" + str(K) + "_" + str(i) + ".txt"
        env = EnvNKlandscape(name_instance)

        print("instance : " + str(i))
        for j in range(nb_restarts):
            print("restart : " + str(j))
            env.reset()
            terminated = False
            current_score = env.score()
            bestScore = float("-inf")

            while not terminated:
                # Extraction des observations à partir de l'état
                neigh = []

                for i in range(N):
                    copied_env = copy.deepcopy(env)
                    obs, deltaFitness, terminated, = copied_env.step(i)
                    neigh.append(deltaFitness)

                if type_strategy == "NN":
                    # Stratégie basée sur le réseau neuronal (NN)
                    non_ordered_neigh = copy.deepcopy(neigh)
                    neigh.sort(reverse=True)
                    neigh = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0)
                    out = network(neigh.unsqueeze(0)).squeeze(0)
                    action = out.argmax().item()
                    action_id = non_ordered_neigh.index(neigh[0][action])
                elif type_strategy == "hillClimber":
                    # Stratégie HillClimber
                    action_id = int(np.argmax(np.array(neigh)))

                elif type_strategy == "IteratedhillClimber":
                    # Stratégie HillClimber itératif
                    if max(neigh) > 0:
                        action_id = int(np.argmax(np.array(neigh)))
                    else:
                        action_id = -1

                if action_id >= 0:
                    state, deltaScore, terminated = env.step(action_id)
                    current_score += deltaScore

                # action de réaliser une perturbation
                elif action_id == -1:
                    env.perturbation(alpha)
                    current_score = env.score()

                if current_score > bestScore:
                    bestScore = current_score

            average_score += bestScore

    return average_score / (nb_instances * nb_restarts)

# Fonction d'évaluation pour CMA-ES
def evaluate_weights_NN(N, K, solution, network, path, nb_instances, nb_restarts):
    # Assurez-vous que la taille du vecteur solution correspond à num_params
    assert len(solution) == num_params

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
    average_score = get_average_score_strategy("NN", N, K, weights, network, path, nb_instances, nb_restarts)

    return average_score

# Paramètres initiaux
N = 64
K = 12
train_path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/" + "train/"
valid_path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/" + "validation/"
nb_restarts = 5
nb_instances = 10
alpha = 0.1  # Calcul de alpha

print("Évaluation de la stratégie HillClimber")
average_score_hillclimber = get_average_score_strategy("hillClimber", N, K, None, None, valid_path, nb_instances, nb_restarts)
print("Score moyen de la stratégie HillClimber sur l'ensemble de validation :")
print(average_score_hillclimber)

# Évaluation de la stratégie IteratedhillClimber
print("Évaluation de la stratégie IteratedhillClimber")
average_score_iterated_hillclimber = get_average_score_strategy("IteratedhillClimber", N, K, None, None, valid_path, nb_instances, nb_restarts, alpha)  # Ajoutez alpha ici
print("Score moyen de la stratégie IteratedhillClimber sur l'ensemble de validation :")
print(average_score_iterated_hillclimber)


# Création de l'architecture du réseau de neurones
layers_size = [N, 2 * N, N]

# Initialisation du réseau de neurones
nnet = Net(layers_size)
params_net = list(nnet.parameters())

# Paramètres CMA-ES
sigma_init = 0.2  # Écart-type initial
num_params = sum(p.numel() for p in params_net)  # Nombre total de paramètres dans le réseau de neurones

# Initialisation de CMA-ES avec un vecteur de poids initial aléatoire
initial_solution = np.random.randn(num_params)
es = cma.CMAEvolutionStrategy(initial_solution, sigma_init)

# Nombre de générations dans CMA-ES
max_generations = 10000

print("Taille de la population dans CMA-ES :", es.popsize)
nb_jobs = es.popsize

# Création d'une liste de réseaux de neurones pour chaque membre de la population CMA-ES
list_nnet = [Net(layers_size) for i in range(es.popsize)]

# Créez une barre de progression avec tqdm
pbar = tqdm(total=max_generations)

# Initialisez la meilleure récompense à un score initial bas (ou négatif)
best_reward = float("-inf")

for generation in range(max_generations):
    solutions = es.ask()  # Échantillonnez de nouveaux vecteurs de poids

    # Évaluez les performances de chaque solution en parallèle
    training_scores = Parallel(n_jobs=nb_jobs)(delayed(evaluate_weights_NN)(N, K, solution, list_nnet[idx], train_path, nb_instances, nb_restarts)  for idx, solution in enumerate(solutions))

    # Mettez à jour la meilleure solution trouvée par CMA-ES
    for idx, reward in enumerate(training_scores):
        if reward > best_reward:
            best_reward = reward
            best_solution = solutions[idx]

    # Mettez à jour CMA-ES avec les performances
    es.tell(solutions, -np.array(training_scores))

    # Évaluez la meilleure solution sur l'ensemble de validation
    validation_score = evaluate_weights_NN(N, K, best_solution, list_nnet[0], valid_path, nb_instances, nb_restarts)
    print("Score moyen sur l'ensemble de validation : " + str(validation_score))

    # Mettez à jour la barre de progression
    pbar.set_postfix(avg_training_score=max(training_scores), avg_validation_score=validation_score)
    pbar.update(1)  # Incrémentation de la barre de progression