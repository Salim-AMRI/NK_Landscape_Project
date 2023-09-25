import os
import matplotlib.pyplot as plt
import numpy as np

# Dossier contenant les fichiers .txt
dossier = './Res'

# Liste pour stocker les colonnes de données extraites de chaque fichier
matrice = []

# Parcourir tous les fichiers dans le dossier
for nom_fichier in os.listdir(dossier):
    if nom_fichier.endswith('.txt'):
        chemin_fichier = os.path.join(dossier, nom_fichier)

        # Ouvrir le fichier en mode lecture
        with open(chemin_fichier, 'r') as fichier:
            # Créer une liste pour stocker les données extraites de ce fichier
            colonne = []

            # Lire les lignes du fichier
            lignes = fichier.readlines()

            # Parcourir chaque ligne et extraire la valeur après la 2e virgule
            for ligne in lignes:
                # Diviser la ligne en utilisant la virgule comme séparateur
                colonnes = ligne.strip().split(',')

                # Vérifier que la ligne contient au moins trois colonnes
                if len(colonnes) >= 3:
                    # Extraire la valeur après la 2e virgule (indice 2)
                    valeur_apres_deuxieme_virgule = colonnes[2].strip()

                    # Assurez-vous que la valeur est un nombre avant de l'ajouter à la colonne
                    try:
                        valeur_reelle = float(valeur_apres_deuxieme_virgule)
                        colonne.append(valeur_reelle)
                    except ValueError:
                        # Gérer le cas où la valeur n'est pas un nombre
                        pass

            # Vérifier que la colonne ne contient pas de NaN
            if not any(np.isnan(colonne)):
                # Ajouter la colonne de données extraites de ce fichier à la matrice
                matrice.append(colonne)

# Vérifier la longueur maximale des colonnes
longueur_maximale = max(len(colonne) for colonne in matrice)

# Calculer la moyenne de chaque ligne en utilisant la somme des valeurs valides
matrice_moyennes = []

for i in range(longueur_maximale):
    ligne_moyennes = [colonne[i] for colonne in matrice if i < len(colonne)]
    valeurs_valides = [valeur for valeur in ligne_moyennes if not np.isnan(valeur)]

    if valeurs_valides:
        moyenne_ligne = sum(valeurs_valides) / len(valeurs_valides)
    else:
        moyenne_ligne = np.nan  # Si aucune valeur valide n'est trouvée
    matrice_moyennes.append(moyenne_ligne)

# Calculer l'écart type de chaque ligne en utilisant les valeurs valides
matrice_ecart_types = []

for i in range(longueur_maximale):
    ligne_ecart_types = [colonne[i] for colonne in matrice if i < len(colonne)]
    valeurs_valides = [valeur for valeur in ligne_ecart_types if not np.isnan(valeur)]

    if len(valeurs_valides) > 1:  # Il faut au moins 2 valeurs pour calculer l'écart type
        ecart_type_ligne = np.std(valeurs_valides, ddof=1)  # ddof=1 pour calculer l'écart type non biaisé
    else:
        ecart_type_ligne = np.nan  # Si moins de 2 valeurs valides sont trouvées
    matrice_ecart_types.append(ecart_type_ligne)

# Générer un nom de fichier incrémental pour le fichier de sortie
numero_fichier_output = 1
nom_fichier_output = f'./results/output_{numero_fichier_output}.txt'
while os.path.isfile(nom_fichier_output):
    numero_fichier_output += 1
    nom_fichier_output = f'./results/output_{numero_fichier_output}.txt'

# Écrire la matrice avec les colonnes de moyennes et d'écart types dans le fichier de sortie
with open(nom_fichier_output, 'w') as fichier_sortie:
    for i in range(longueur_maximale):
        # Construire la ligne formatée en utilisant les indices
        ligne_formattee = '\t'.join([str(colonne[i]) for colonne in matrice if i < len(colonne)])
        ligne_formattee += f"\t{matrice_moyennes[i]}\t{matrice_ecart_types[i]}"  # Ajouter la moyenne et l'écart type à la fin de chaque ligne
        fichier_sortie.write(f"{ligne_formattee}\n")

# Chargement des données depuis le fichier CSV 'hillClimber_results_1.csv' et calcul de la moyenne
chemin_fichier_csv = './results/hillClimber_results_1.csv'
valeurs_hill_climber = []

with open(chemin_fichier_csv, 'r') as fichier_csv:
    lignes_csv = fichier_csv.readlines()

    # Vérifiez s'il y a au moins deux lignes dans le fichier CSV
    if len(lignes_csv) >= 2:
        # Ignorez la première ligne (en-tête) et lisez les données à partir de la deuxième ligne
        for ligne_csv in lignes_csv[1:]:
            colonnes_csv = ligne_csv.strip().split(',')
            if len(colonnes_csv) >= 2:  # Vérifiez qu'il y a au moins 2 colonnes (pour éviter les erreurs)
                valeur_apres_premiere_virgule = float(colonnes_csv[1])  # Récupérez la valeur après la première virgule
                valeurs_hill_climber.append(valeur_apres_premiere_virgule)

# Calcul de la moyenne des valeurs de hillClimber_results_1.csv
if valeurs_hill_climber:
    moyenne_hill_climber = np.mean(valeurs_hill_climber)
else:
    moyenne_hill_climber = 0  # Moyenne par défaut si aucune valeur n'est trouvée

# Créez une liste avec la moyenne calculée de la même longueur que matrice_moyennes
longueur_maximale = max(len(colonne) for colonne in matrice)
moyenne_hill_climber_liste = [moyenne_hill_climber] * longueur_maximale

# Créer une liste d'indices (numéro de ligne) pour l'axe x
indices = list(range(1, len(matrice_moyennes) + 1))

# Tracer le graphique de la moyenne avec la zone d'écart type
plt.figure(figsize=(10, 6))
#plt.plot(indices, matrice_moyennes, marker='o', linestyle='-', label='Moyenne')
plt.plot(indices, matrice_moyennes, linestyle='-', label='Moyenne')  # Ligne continue pour la moyenne
#plt.fill_between(indices, [m + s for m, s in zip(matrice_moyennes, matrice_ecart_types)], [m - s for m, s in zip(matrice_moyennes, matrice_ecart_types)], alpha=0.2, label='Écart Type')
plt.fill_between(indices, [m + s for m, s in zip(matrice_moyennes, matrice_ecart_types)], [m - s for m, s in zip(matrice_moyennes, matrice_ecart_types)], alpha=0.2, linestyle='-', label='Écart Type')  # Zone continue pour l'écart type
plt.xlabel('Nombre de générations')
plt.ylabel('Moyenne / Écart Type')
plt.title('Évolution de la moyenne et de l\'écart type par rapport au nombre de générations')
plt.legend()
plt.grid(True)

# Tracer la droite de la moyenne hillClimber
plt.plot(indices, moyenne_hill_climber_liste[:longueur_maximale], linestyle='-', label='Moyenne Hill Climber')
plt.legend()

# Générer un nom de fichier incrémental pour le graphique
numero_fichier_graphique = 1
nom_fichier_graphique = f'graphique_evolution_{numero_fichier_graphique}.png'
while os.path.isfile(nom_fichier_graphique):
    numero_fichier_graphique += 1
    nom_fichier_graphique = f'graphique_evolution_{numero_fichier_graphique}.png'

# Sauvegarder le graphique dans un fichier image incrémental
plt.savefig("./results/" + nom_fichier_graphique)

# Afficher le graphique
plt.show()