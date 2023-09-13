import numpy as np

class Nk_generator():

    def __init__(self, n, k, instances, folder="/home/etud/Téléchargements/Generate/"):
        # Initialisation de la classe avec les paramètres n, k, instances et folder
        self.N = n
        self.K = k
        self.instances = instances
        self.folder = folder

        # Pour chaque instance, créer un fichier et y écrire les données
        for i in range(self.instances):
            fichier = open(self.folder + "nk_" + str(self.N) + "_" + str(self.K) + "_" + str(i) + ".txt", "w")
            fichier.write(str(self.N) + " " + str(self.K))  # Écrit la taille N et la complexité K en début de fichier

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
            fichier.close()  # Ferme le fichier

# Création d'une instance de Nk_generator avec des valeurs spécifiques
Nk_generator(64, 12, 10)
