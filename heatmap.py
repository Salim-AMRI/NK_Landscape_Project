import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Chemin du répertoire contenant les fichiers journaux
log_directory = 'log_trajectory'

# Assurez-vous que le répertoire de journalisation existe
if not os.path.exists(log_directory):
    print(f"Le répertoire {log_directory} n'existe pas.")
    exit()

# Créez des listes pour stocker les données (Actions Above Chosen et Positive Actions)
positive_actions = []
actions_above_chosen = []

# Parcourez les fichiers journaux dans le répertoire
for filename in os.listdir(log_directory):
    if "logTrajectory_strategyNN_N_64_K_8" in filename:
        # Lisez le contenu du fichier journal
        with open(os.path.join(log_directory, filename), 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                if "Positive Actions:" in line and "Actions Above Chosen:" in line:
                    # Utilisez des expressions régulières pour extraire les valeurs numériques
                    positive_count = int(re.search(r"Positive Actions: (\d+)", line).group(1))
                    actions_above_count = int(re.search(r"Actions Above Chosen: (\d+)", line).group(1))

                    # Ajoutez les valeurs extraites à vos listes
                    positive_actions.append(positive_count)
                    actions_above_chosen.append(actions_above_count)

# Convertissez les listes en tableaux NumPy pour créer le heatmap
positive_actions = np.array(positive_actions)
actions_above_chosen = np.array(actions_above_chosen)

# Créez un heatmap en utilisant Matplotlib
plt.figure(figsize=(10, 6))
plt.hist2d(actions_above_chosen, positive_actions, bins=(20, 20), cmap=plt.cm.Blues)
plt.xlabel("Actions Above Chosen")
plt.ylabel("Positive Actions")
plt.title("Heatmap des Actions Above Chosen vs Positive Actions")
plt.colorbar()

# Enregistrez le graphique de manière incrémentielle
num = 1
filename = f'./images/heatmap_{num}.png'
while os.path.isfile(filename):
    num += 1
    filename = f'./images/heatmap_{num}.png'

plt.savefig(filename)

plt.show()
