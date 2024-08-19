import os
import re
import matplotlib.pyplot as plt

# Définissez le chemin du dossier contenant les fichiers journaux
log_directory = 'log_trajectory'

# Définissez le nom du fichier journal que vous souhaitez visualiser




#log_filename = 'logTrajectory_strategyNN_N_64_K_8_nb_instances_test_6_nb_restarts_0.log'


#log_filename = 'logTrajectory_StrategyNNFitness_and_current_N_64_K_8_nb_instances_test_6_nb_restarts_0.log'

#log_filename = 'logTrajectory_StrategyNNRanked_v2_N_64_K_8_nb_instances_test_6_nb_restarts_0.log'
log_filename = 'logTrajectory_StrategyNNRanked_v2_zScore_N_64_K_8_nb_instances_test_6_nb_restarts_0.log'


#log_filename = 'logTrajectory_strategyNN_N_64_K_8_nb_instances_test_6_nb_restarts_0.log'

# Assurez-vous que le répertoire de journalisation existe
if not os.path.exists(log_directory):
    print(f"Le répertoire {log_directory} n'existe pas.")
    exit()

# Créez des listes pour stocker les données d'itération, de score, de positive_count et d'actions_above_count
iterations = []
scores = []
positive_counts = []
actions_above_counts = []

# Lisez le contenu du fichier journal
with open(os.path.join(log_directory, log_filename), 'r', encoding='utf-8', errors='replace') as file:
    for line in file:
        if "Turn:" in line and "Current Score:" in line:
            # Utilisez des expressions régulières pour extraire les valeurs numériques
            iteration = int(re.search(r"Turn: (\d+)", line).group(1))
            score = float(re.search(r"Current Score: ([\d.]+)", line).group(1))

            # Ajoutez les valeurs extraites à vos listes
            iterations.append(iteration)
            scores.append(score)

        if "Positive Actions:" in line and "Actions Above Chosen:" in line:
            # Utilisez des expressions régulières pour extraire les valeurs numériques
            positive_count = int(re.search(r"Positive Actions: (\d+)", line).group(1))+1
            actions_above_count = int(re.search(r"Actions Above Chosen: (\d+)", line).group(1))+1

            # Ajoutez les valeurs extraites à vos listes
            positive_counts.append(positive_count)
            actions_above_counts.append(actions_above_count)

# Créez une figure avec deux sous-graphiques l'un en dessous de l'autre
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Ajoutez le graphique du haut : Évolution du score au fil des générations
ax1.plot(iterations, [score/64 for score in scores], marker='o', linestyle='-')
ax1.set_xlabel("Number of iterations", fontsize=20)
ax1.set_ylabel("Fitness score", fontsize=20)
ax1.set_title("Fitness evolution during a trajectory of Neuro-LS (o4)", fontsize=24)
ax1.grid(True)

# Ajoutez le graphique du bas : Nuage de points pour Positive Count et Actions Above Count
ax2.scatter(iterations, positive_counts, marker='o', s=30, label='Nbr of actions with positive improvement', color='blue')
ax2.scatter(iterations, actions_above_counts, marker='x', s=30, label='Rank of the selected action', color='red')
ax2.set_xlabel("Number of iterations", fontsize=20)
ax2.set_ylabel("Nb actions/Rank", fontsize=20)

ax2.yaxis.set_inverted(True)

ax2.set_title("Neuro-LS strategy (o4)" , fontsize=24)
ax2.legend(fontsize=20)
ax2.grid(True)

# Inversez l'axe Y du graphique du bas
#ax2.invert_yaxis()

# Ajustez l'espacement entre les sous-graphiques
plt.tight_layout()

# Affichez la figure avec les sous-graphiques
plt.show()
