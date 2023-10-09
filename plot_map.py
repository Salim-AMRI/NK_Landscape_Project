import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Définissez le répertoire où se trouvent vos fichiers .log
log_directory = "log_trajectory"

# Définissez une expression régulière pour filtrer les fichiers .log correspondants
log_filename_pattern = r"logTrajectory_strategyNN_N_32_K_1_(\d+)\.log"

# Créez une liste pour stocker les noms de fichiers correspondants
matching_log_files = []

# Parcourez les fichiers du répertoire
for filename in os.listdir(log_directory):
    match = re.match(log_filename_pattern, filename)
    if match:
        matching_log_files.append(os.path.join(log_directory, filename))

# Maintenant, vous avez une liste de fichiers correspondants que vous pouvez traiter
for data_file in matching_log_files:
    # Lisez le fichier de log et effectuez le traitement nécessaire ici
    with open(data_file, 'r') as file:
        lines = file.readlines()  # Lisez les lignes du fichier ici

    # Initialisez les compteurs pour les données du heatmap
    max_positive_actions = 0
    max_actions_above_chosen = 0

    # Parcourez chaque ligne du fichier pour extraire les données
    for line in lines:
        match = re.match(r".*Positive Actions: (\d+), Actions Above Chosen: (\d+).*", line)
        if match:
            positive_actions = int(match.group(1))
            actions_above_chosen = int(match.group(2))
            max_positive_actions = max(max_positive_actions, positive_actions)
            max_actions_above_chosen = max(max_actions_above_chosen, actions_above_chosen)

    # Créez une matrice vide pour stocker les données du heatmap
    heatmap_data = np.zeros((max_positive_actions + 1, max_actions_above_chosen + 1))

    # Parcourez à nouveau le fichier pour remplir la matrice du heatmap
    for line in lines:
        match = re.match(r".*Positive Actions: (\d+), Actions Above Chosen: (\d+).*Action Contribution: (\d+\.\d+).*", line)
        if match:
            positive_actions = int(match.group(1))
            actions_above_chosen = int(match.group(2))
            action_contribution = float(match.group(3))
            heatmap_data[positive_actions][actions_above_chosen] += action_contribution

    # Configurez l'affichage de la heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap_data.T, origin='lower', extent=[0, max_positive_actions, 0, max_actions_above_chosen], aspect='auto', cmap='viridis')
    plt.colorbar(label="Contribution de l'action")
    plt.xlabel('Rang de la solution sélectionnée')
    plt.ylabel('Nombre de voisins dont la DeltaFitness est positive')
    plt.title('Heatmap des contributions d\'action')
    plt.grid(False)

    # Affichez la heatmap
    plt.show()
