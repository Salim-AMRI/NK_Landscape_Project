import os

# Chemin du répertoire contenant les fichiers à consulter
directory = 'results_09102023/'

# Initialisation de la liste des valeurs maximales
max_values = []

# Parcours des fichiers dans le répertoire
for filename in os.listdir(directory):
    if filename.endswith('.txt') and 'test_strategy_strategyNN_10,5_N_32_K_8' in filename:
        file_path = os.path.join(directory, filename)

        # Initialisation de la valeur maximale pour ce fichier
        max_value = None

        # Ouvre le fichier en mode lecture
        with open(file_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                # Sépare la ligne en éléments en utilisant la virgule comme séparateur
                elements = line.strip().split(',')

                # Vérifie s'il y a au moins 3 éléments (après la deuxième virgule)
                if len(elements) >= 3:
                    try:
                        # Convertit le troisième élément en un nombre flottant
                        value = float(elements[2])

                        # Si max_value est None ou si la valeur est supérieure à max_value, met à jour max_value
                        if max_value is None or value > max_value:
                            max_value = value

                    except ValueError:
                        # En cas d'erreur de conversion, ignore cette ligne
                        pass

        # Ajoute la valeur maximale de ce fichier à la liste des valeurs maximales
        if max_value is not None:
            max_values.append(max_value)

# Calcule la moyenne des valeurs maximales
if max_values:
    average_max_value = sum(max_values) / len(max_values)
    print(f"Moyenne des valeurs maximales : {average_max_value}")
else:
    print("Aucune valeur maximale trouvée dans les fichiers.")
