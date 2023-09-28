import os
import matplotlib.pyplot as plt
import numpy as np

# Dossier contenant les fichiers .txt
dossier = './results'

# Liste pour stocker les colonnes de données extraites de chaque fichier
train_matrice = []
valid_matrice = []
valeurs_hillclimber = []

#### Valid_NN

# Parcourir tous les fichiers dans le dossier
for nom_fichier in os.listdir(dossier):
    if nom_fichier.endswith('.txt') and 'NN' in nom_fichier:
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
                valid_matrice.append(colonne)

# Vérifier la longueur maximale des colonnes
longueur_maximale = max(len(colonne) for colonne in valid_matrice)

# Calculer la moyenne de chaque ligne en utilisant la somme des valeurs valides
matrice_moyennes = []

for i in range(longueur_maximale):
    ligne_moyennes = [colonne[i] for colonne in valid_matrice if i < len(colonne)]
    valeurs_valides = [valeur for valeur in ligne_moyennes if not np.isnan(valeur)]

    if valeurs_valides:
        moyenne_ligne = sum(valeurs_valides) / len(valeurs_valides)
    else:
        moyenne_ligne = np.nan  # Si aucune valeur valide n'est trouvée
    matrice_moyennes.append(moyenne_ligne)

# Calculer l'écart type de chaque ligne en utilisant les valeurs valides
matrice_ecart_types = []

for i in range(longueur_maximale):
    ligne_ecart_types = [colonne[i] for colonne in valid_matrice if i < len(colonne)]
    valeurs_valides = [valeur for valeur in ligne_ecart_types if not np.isnan(valeur)]

    if len(valeurs_valides) > 1:  # Il faut au moins 2 valeurs pour calculer l'écart type
        ecart_type_ligne = np.std(valeurs_valides, ddof=1)  # ddof=1 pour calculer l'écart type non biaisé
    else:
        ecart_type_ligne = np.nan  # Si moins de 2 valeurs valides sont trouvées
    matrice_ecart_types.append(ecart_type_ligne)

# Générer un nom de fichier incrémental pour le fichier de sortie
numero_fichier_output = 1
nom_fichier_output = f'./Res/output_valid_{numero_fichier_output}.txt'
while os.path.isfile(nom_fichier_output):
    numero_fichier_output += 1
    nom_fichier_output = f'./Res/output_valid_{numero_fichier_output}.txt'

# Écrire la matrice avec les colonnes de moyennes et d'écart types dans le fichier de sortie
with open(nom_fichier_output, 'w') as fichier_sortie:
    for i in range(longueur_maximale):
        # Construire la ligne formatée en utilisant les indices
        ligne_formattee = '\t'.join([str(colonne[i]) for colonne in valid_matrice if i < len(colonne)])
        ligne_formattee += f"\t{matrice_moyennes[i]}\t{matrice_ecart_types[i]}"  # Ajouter la moyenne et l'écart type à la fin de chaque ligne
        fichier_sortie.write(f"{ligne_formattee}\n")

#### train_NN

# Parcourir tous les fichiers dans le dossier
for nom_fichier in os.listdir(dossier):
    if nom_fichier.endswith('.txt') and 'NN' in nom_fichier:
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
                    valeur_apres_premiere_virgule = colonnes[1].strip()

                    # Assurez-vous que la valeur est un nombre avant de l'ajouter à la colonne
                    try:
                        valeur_reelle = float(valeur_apres_premiere_virgule)
                        colonne.append(valeur_reelle)
                    except ValueError:
                        # Gérer le cas où la valeur n'est pas un nombre
                        pass

            # Vérifier que la colonne ne contient pas de NaN
            if not any(np.isnan(colonne)):
                # Ajouter la colonne de données extraites de ce fichier à la matrice
                train_matrice.append(colonne)

# Vérifier la longueur maximale des colonnes
longueur_maximale = max(len(colonne) for colonne in valid_matrice)

# Calculer la moyenne de chaque ligne en utilisant la somme des valeurs valides
matrice_moyennes_train = []

for i in range(longueur_maximale):
    ligne_moyennes = [colonne[i] for colonne in train_matrice if i < len(colonne)]
    valeurs_valides = [valeur for valeur in ligne_moyennes if not np.isnan(valeur)]

    if valeurs_valides:
        moyenne_ligne = sum(valeurs_valides) / len(valeurs_valides)
    else:
        moyenne_ligne = np.nan  # Si aucune valeur valide n'est trouvée
    matrice_moyennes_train.append(moyenne_ligne)

# Calculer l'écart type de chaque ligne en utilisant les valeurs valides
matrice_ecart_types_train = []

for i in range(longueur_maximale):
    ligne_ecart_types = [colonne[i] for colonne in train_matrice if i < len(colonne)]
    valeurs_valides = [valeur for valeur in ligne_ecart_types if not np.isnan(valeur)]

    if len(valeurs_valides) > 1:  # Il faut au moins 2 valeurs pour calculer l'écart type
        ecart_type_ligne = np.std(valeurs_valides, ddof=1)  # ddof=1 pour calculer l'écart type non biaisé
    else:
        ecart_type_ligne = np.nan  # Si moins de 2 valeurs valides sont trouvées
    matrice_ecart_types_train.append(ecart_type_ligne)

# Générer un nom de fichier incrémental pour le fichier de sortie
numero_fichier_output = 1
nom_fichier_output = f'./Res/output_train_{numero_fichier_output}.txt'
while os.path.isfile(nom_fichier_output):
    numero_fichier_output += 1
    nom_fichier_output = f'./Res/output_train_{numero_fichier_output}.txt'

# Écrire la matrice avec les colonnes de moyennes et d'écart types dans le fichier de sortie
with open(nom_fichier_output, 'w') as fichier_sortie:
    for i in range(longueur_maximale):
        # Construire la ligne formatée en utilisant les indices
        ligne_formattee = '\t'.join([str(colonne[i]) for colonne in valid_matrice if i < len(colonne)])
        ligne_formattee += f"\t{matrice_moyennes[i]}\t{matrice_ecart_types[i]}"  # Ajouter la moyenne et l'écart type à la fin de chaque ligne
        fichier_sortie.write(f"{ligne_formattee}\n")

############ hillclimber

# Parcourir tous les fichiers dans le dossier
for nom_fichier in os.listdir(dossier):
    if nom_fichier.endswith('.txt') and 'hillClimber' in nom_fichier:
        chemin_fichier = os.path.join(dossier, nom_fichier)

        # Ouvrir le fichier en mode lecture
        with open(chemin_fichier, 'r') as fichier:
            # Créer une liste pour stocker les données extraites de ce fichier
            colonne = []

            # Lire les lignes du fichier
            lignes = fichier.readlines()

            # Parcourir chaque ligne et extraire la valeur après la 2e virgule de la 2e ligne
            if len(lignes) > 1:
                deuxieme_ligne = lignes[1].strip()
                deuxieme_ligne_elements = deuxieme_ligne.split(',')
                if len(deuxieme_ligne_elements) > 2:
                    valeur_hillclimber = deuxieme_ligne_elements[2].strip()
                    try:
                        valeur_reelle = float(valeur_hillclimber)
                        valeurs_hillclimber.append(valeur_reelle)
                    except ValueError:
                        pass

# Calculer la moyenne des valeurs hillClimber
moyenne_hillclimber = np.nanmean(valeurs_hillclimber)
# Calculer l'écart type des valeurs hillClimber
ecart_type_hillclimber = np.nanstd(valeurs_hillclimber, ddof=1)

# Générer un nom de fichier incrémental pour le fichier .txt
numero_fichier_txt = 1
nom_fichier_txt = f'hillclimber_moyenne_{numero_fichier_txt}.txt'
while os.path.isfile(os.path.join("./Res", nom_fichier_txt)):
    numero_fichier_txt += 1
    nom_fichier_txt = f'hillclimber_moyenne_{numero_fichier_txt}.txt'

# Écrire la moyenne hillClimber dans le fichier .txt
with open(os.path.join("./Res", nom_fichier_txt), 'w') as fichier_txt:
    fichier_txt.write(f"Moyenne HillClimber : {moyenne_hillclimber}\n")
    fichier_txt.write(f"Écart Type HillClimber : {ecart_type_hillclimber}\n")

### Graphique

# Créer une liste d'indices (numéro de ligne) pour l'axe x
indices = list(range(1, len(matrice_moyennes) + 1))

# Tracer le graphique de la moyenne avec la zone d'écart type
plt.figure(figsize=(10, 6))
plt.xlim(0, max(indices))
plt.axhline(y=moyenne_hillclimber, color='r', linestyle='-', label='HC_Moyenne')  # Droite horizontale pour la moyenne hillClimber
plt.fill_between(indices, [moyenne_hillclimber + ecart_type_hillclimber] * len(indices), [moyenne_hillclimber - ecart_type_hillclimber] * len(indices), color='r', alpha=0.2, label='HC_Écart_Type')  # Zone continue pour l'écart type hillClimber
plt.plot(indices, matrice_moyennes_train, linestyle='-', label='T_Moyenne')  # Ligne continue pour la moyenne
plt.fill_between(indices, [m + s for m, s in zip(matrice_moyennes_train, matrice_ecart_types_train)], [m - s for m, s in zip(matrice_moyennes_train, matrice_ecart_types_train)], alpha=0.2, linestyle='-', label='T_Écart_Type')  # Zone continue pour l'écart type
plt.plot(indices, matrice_moyennes, linestyle='-', label='V_Moyenne')  # Ligne continue pour la moyenne
plt.fill_between(indices, [m + s for m, s in zip(matrice_moyennes, matrice_ecart_types)], [m - s for m, s in zip(matrice_moyennes, matrice_ecart_types)], alpha=0.2, linestyle='-', label='V_Écart_Type')  # Zone continue pour l'écart type
plt.xlabel('Nombre de générations')
plt.ylabel('Moyenne / Écart Type')
plt.title('Évolution de la moyenne et de l\'écart type par rapport au nombre de générations')
plt.legend()
plt.grid(True)

# Générer un nom de fichier incrémental pour le graphique
numero_fichier_graphique = 1
nom_fichier_graphique = f'graphique_evolution_{numero_fichier_graphique}.png'
while os.path.isfile(os.path.join("./Res", nom_fichier_graphique)):
    numero_fichier_graphique += 1
    nom_fichier_graphique = f'graphique_evolution_{numero_fichier_graphique}.png'

# Sauvegarder le graphique dans un fichier image incrémental
plt.savefig(os.path.join("./Res", nom_fichier_graphique))

# Afficher le graphique
plt.show()
