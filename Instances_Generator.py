import numpy as np
import os
import shutil  # Importez le module 'shutil' pour la suppression de fichiers et de dossiers

class Nk_generator():

    def __init__(self, n, k, instances, base_folder):
        # Initialisation de la classe avec les paramètres n, k, instances et le dossier de base
        self.N = n
        self.K = k
        self.instances = instances
        self.base_folder = base_folder

        # Construisez le chemin complet du dossier de train en fonction de N et K
        #self.train_folder = os.path.join(base_folder, f'NK_{self.N}_{self.K}')

        self.train_folder = base_folder

        # Assurez-vous que le dossier de train existe, sinon, créez-le
        os.makedirs(self.train_folder, exist_ok=True)

        # Supprimez le contenu du dossier de train existant s'il y en a
        #self.clean_train_folder()

        # Pour chaque instance, créer un fichier et y écrire les données
        for i in range(self.instances):

            print(self.train_folder)
            fichier = open(os.path.join(self.train_folder, f"nk_{self.N}_{self.K}_{i}.txt"), "w")
            fichier.write(str(self.N) + " " + str(self.K))

            # Générer les voisins de chaque élément dans le paysage NK
            for x in range(self.N):
                neigh = []
                neigh.append(x)
                for y in range(self.K):
                    x1 = np.random.randint(0, self.N)
                    while x1 in neigh:
                        x1 = np.random.randint(0, self.N)
                    neigh.append(x1)

                neigh.sort()  # Trie les voisins pour assurer un ordre
                for x2 in neigh:
                    fichier.write("\n" + str(x2))  # Écrit les voisins dans le fichier

            # Générer des valeurs aléatoires pour chaque élément dans le paysage NK
            for x in range(self.N):
                for y in range(2 ** (self.K + 1)):
                    fichier.write(
                        "\n" + str(round(np.random.random(), 6)))  # Écrit des valeurs aléatoires dans le fichier
            fichier.close()

    # Supprimez le contenu du dossier de train existant s'il y en a
    def clean_train_folder(self):
        if os.path.exists(self.train_folder):
            shutil.rmtree(self.train_folder)
