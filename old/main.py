import numpy as np
import torch
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.nn as nn
import cma
import multiprocessing

class EnvNKlandscape:
    def __init__(self, file):
        f = open(file, "r")
        lignes = f.readlines()

        head = lignes[0].split()
        self.N = int(head[0])
        self.K = int(head[1])

        self.links = []
        for n in range(self.N):
            self.links.append([])
            for k in range(self.K + 1):
                self.links[n].append(int(lignes[1 + n * (self.K + 1) + k]))
            self.links[n].append([])
            for k in range(2 ** (self.K + 1)):
                self.links[n][-1].append(float(lignes[1 + self.N * (self.K + 1) + n * (2 ** (self.K + 1)) + k]))

        f.close()

        self.max_nb_turn = 2 * self.N

        self.game_state = np.random.randint(2, size=self.N)

        self.turn = 0

    def reset(self):
        self.game_state = np.random.randint(2, size=self.N)
        self.turn = 0
        return self.game_state

    def step(self, action):
        old_value = self.game_state[action]
        self.game_state[action] = (self.game_state[action] + 1) % 2
        deltaFitness = 0

        for link in self.links:
            if action in link:
                malus = []
                bonus = []
                for i in link[:-1]:
                    bonus.append(self.game_state[i])
                    if i == action:
                        malus.append(old_value)
                    else:
                        malus.append(self.game_state[i])
                malus_index = 0
                bonus_index = 0
                for i in range(self.K + 1):
                    malus_index += (2 ** (self.K - i)) * malus[i]
                    bonus_index += (2 ** (self.K - i)) * bonus[i]

                deltaFitness -= link[-1][int(malus_index)]
                deltaFitness += link[-1][int(bonus_index)]

        self.turn += 1

        if self.turn == self.max_nb_turn:
            terminated = True
        else:
            terminated = False

        return self.game_state, deltaFitness, terminated

    def score(self):
        sco = 0

        for link in self.links:
            bonus = []
            for i in link[:-1]:
                bonus.append(self.game_state[i])
            bonus_index = 0
            for i in range(self.K + 1):
                bonus_index += (2 ** (self.K - i)) * bonus[i]
            sco += link[-1][int(bonus_index)]

        return sco

class Net(nn.Module):
    def __init__(self, layers_size):
        super(Net, self).__init__()
        self.layers_size = layers_size
        layers = []

        for i in range(len(layers_size) - 1):
            layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))
            if i != len(layers_size) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

def get_total_reward_trajectory(N, weights, model, env):
    cloned_model = copy.deepcopy(model)

    with torch.no_grad():
        for param, weight in zip(cloned_model.parameters(), weights):
            param.copy_(weight)

    state = env.reset()
    terminated = False
    total_reward = 0.0

    while not terminated:
        neigh = []

        for i in range(N):
            copied_env = copy.deepcopy(env)
            obs, deltaFitness, terminated, = copied_env.step(i)
            neigh.append(deltaFitness)

        non_ordered_neigh = copy.deepcopy(neigh)
        neigh.sort(reverse=True)

        neigh = torch.tensor(neigh, dtype=torch.float32).unsqueeze(0)

        out = cloned_model(neigh.unsqueeze(0)).squeeze(0)

        action = out.argmax().item()
        action_id = non_ordered_neigh.index(neigh[0][action])

        state, deltaFitness, terminated = env.step(action_id)

        total_reward += deltaFitness

    return total_reward

def objective_function(weights, env):
    total_reward = get_total_reward_trajectory(N, weights, nnet, env)
    return total_reward

N = 64
K = 1
nb_generations = 10
nb_jobs = 20
size_pop = 200
nb_offsprings = 20

instance_files = [
    "nk_64_1_0.txt",
    "nk_64_1_1.txt",
    "nk_64_1_2.txt",
    "nk_64_1_3.txt",
    "nk_64_1_4.txt",
    "nk_64_1_0.txt",
    "nk_64_1_1.txt",
    "nk_64_1_2.txt",
    "nk_64_1_3.txt",
    "nk_64_1_4.txt",
]

all_results = []

# Définir nnet en dehors de la boucle
layers_size = [N, 4 * N, 2 * N, N]
nnet = Net(layers_size)
params_net = list(nnet.parameters())

def process_instance(instance_file):
    try:
        env = EnvNKlandscape(instance_file)
    except Exception as e:
        print(f"Erreur lors de l'initialisation de l'environnement pour {instance_file}: {str(e)}")
        return {'instance_file': instance_file, 'best_score': 0}  # Retourner un score nul en cas d'erreur

    x0 = np.random.rand(N)
    initial_sigma = 0.1

    es = cma.CMAEvolutionStrategy(x0, initial_sigma)

    best_score = 0

    for generation in range(nb_generations):
        solutions = es.ask()
        deltaFitness = [objective_function(solution, env) for solution in solutions]  # Passer env comme argument
        es.tell(solutions, deltaFitness)
        current_score = np.max(deltaFitness)
        if current_score > best_score:
            best_score = current_score

    # Ajouter les résultats de cette instance à la liste globale
    all_results.append({'instance_file': instance_file, 'best_score': best_score})

# Créer une pool de processus
with multiprocessing.Pool(processes=nb_jobs) as pool:
    # Exécuter les instances en parallèle
    list(tqdm(pool.map(process_instance, instance_files), total=len(instance_files)))

# Calculer la moyenne des meilleurs scores pour l'ensemble des instances
total_best_scores = [result['best_score'] for result in all_results]
average_best_score = sum(total_best_scores) / len(all_results)

# Afficher la moyenne des meilleurs scores pour l'ensemble des instances
print("Moyenne des meilleurs scores pour toutes les instances :", average_best_score)


parser.add_argument('--verbose', action='store_true', help='Afficher des informations de progression')

# Exécutez le script avec des arguments de ligne de commande :
python /home/etud/TestProjet/NKL_HC.py 64 1 --nb-restarts 5 --max-generations 10000 --verbose