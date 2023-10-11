import os
import re
import matplotlib.pyplot as plt

# Définissez le chemin du dossier contenant les fichiers journaux
log_directory = 'log_trajectory'

# Définissez le nom du fichier journal que vous souhaitez visualiser
log_filename = 'logTrajectory_strategyNN_N_64_K_8_nb_instances_test_0_nb_restarts_0.log'

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
            positive_count = int(re.search(r"Positive Actions: (\d+)", line).group(1))
            actions_above_count = int(re.search(r"Actions Above Chosen: (\d+)", line).group(1))

            # Ajoutez les valeurs extraites à vos listes
            positive_counts.append(positive_count)
            actions_above_counts.append(actions_above_count)

# Créez une figure avec deux sous-graphiques l'un en dessous de l'autre
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Ajoutez le graphique du haut : Évolution du score au fil des générations
ax1.plot(iterations, scores, marker='o', linestyle='-')
ax1.set_xlabel("Nombre d'itérations")
ax1.set_ylabel("Score")
ax1.set_title("Évolution du score au cours des générations")
ax1.grid(True)

# Ajoutez le graphique du bas : Nuage de points pour Positive Count et Actions Above Count
ax2.scatter(iterations, positive_counts, marker='o', s=30, label='Positive Count', color='blue')
ax2.scatter(iterations, actions_above_counts, marker='x', s=30, label='Actions Above Count', color='red')
ax2.set_xlabel("Nombre d'itérations")
ax2.set_ylabel("Valeurs")
ax2.set_title("Nuage de points de Positive Count et Actions Above Count au fil des générations")
ax2.legend()
ax2.grid(True)

# Ajustez l'espacement entre les sous-graphiques
plt.tight_layout()

# Affichez la figure avec les sous-graphiques
plt.show()
