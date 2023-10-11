import os
import matplotlib.pyplot as plt
import numpy as np

# Fonction pour enregistrer les graphiques dans le dossier "Res"
def save_plot_to_res(filename):
    # Assurez-vous que le dossier "Res" existe, sinon, créez-le
    if not os.path.exists("./Res"):
        os.makedirs("./Res")

    # Enregistrez le graphique dans le dossier "Res"
    plt.savefig(os.path.join("./Res", filename))
    print(f"Graphique enregistré sous : ./Res/{filename}")

def extract_data_from_file(file_path, keyword):
    column_data = {'valid': [], 'train': []}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            columns = line.strip().split(',')

            if len(columns) >= 3:
                try:
                    value_valid = float(columns[2].strip())
                    value_train = float(columns[1].strip())

                    if not np.isnan(value_valid):
                        column_data['valid'].append(value_valid)

                    if not np.isnan(value_train):
                        column_data['train'].append(value_train)
                except ValueError:
                    pass

    return column_data

def extract_and_process_data(directory, keywords):
    data_matrices = {'valid': [], 'train': []}
    means_list = {'valid': [], 'train': []}
    std_devs_list = {'valid': [], 'train': []}
    max_list = {'valid': [], 'train': []}
    min_list = {'valid': [], 'train': []}

    for keyword in keywords:
        data_matrix = {'valid': [], 'train': []}

        for filename in os.listdir(directory):
            if filename.endswith('.txt') and keyword in filename:
                file_path = os.path.join(directory, filename)

                column_data = extract_data_from_file(file_path, keyword)

                for data_type in ['valid', 'train']:
                    data = column_data[data_type]

                    if data:
                        data_matrix[data_type].append(data)

        for data_type in ['valid', 'train']:
            if data_matrix[data_type]:
                max_length = max(len(column) for column in data_matrix[data_type])
            else:
                max_length = 0

            means = []
            std_devs = []
            max_values = []
            min_values = []

            for i in range(max_length):
                valid_values = [column[i] for column in data_matrix[data_type] if i < len(column) and not np.isnan(column[i])]

                if valid_values:
                    mean = sum(valid_values) / len(valid_values)
                    max_val = max(valid_values)
                    min_val = min(valid_values)
                else:
                    mean = np.nan
                    max_val = np.nan
                    min_val = np.nan

                means.append(mean)
                max_values.append(max_val)
                min_values.append(min_val)

                if len(valid_values) > 1:
                    std_dev = np.std(valid_values, ddof=1)
                else:
                    std_dev = np.nan

                std_devs.append(std_dev)

            data_matrices[data_type].append(data_matrix[data_type])
            means_list[data_type].append(means)
            std_devs_list[data_type].append(std_devs)
            max_list[data_type].append(max_values)
            min_list[data_type].append(min_values)

    return data_matrices, means_list, std_devs_list, max_list, min_list

def extract_localSearch_data(directory, keywords):
    values_localSearch = []
    for filename in os.listdir(directory):
        for keyword in keywords:
            if filename.endswith('.txt') and keyword in filename:
                file_path = os.path.join(directory, filename)

                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    if len(lines) > 1:
                        second_line = lines[1].strip()
                        second_line_elements = second_line.split(',')

                        if len(second_line_elements) > 2:
                            value_localSearch = second_line_elements[2].strip()

                            try:
                                value_reelle = float(value_localSearch)
                                values_localSearch.append(value_reelle)
                            except ValueError:
                                pass

    if values_localSearch:  # Vérifiez si la liste n'est pas vide
        mean_localSearch = np.nanmean(values_localSearch)
        std_dev_localSearch = np.nanstd(values_localSearch, ddof=1)
    else:
        mean_localSearch = np.nan
        std_dev_localSearch = np.nan

    return mean_localSearch, std_dev_localSearch


def plot_data_evolution(indices, data, localSearch_data, max_list, min_list, save_filename=None):
    plt.figure(figsize=(10, 6))
    plt.xlim(0, max(indices))

    # Définissez une palette de couleurs distincte pour chaque mot-clé
    cmap = plt.get_cmap('tab10')
    num_colors = len(localSearch_keywords)
    colors = [cmap(i) for i in range(num_colors)]

    data_types = ['valid', 'train']

    for data_type in data_types:
        for i, (means, std_devs) in enumerate(zip(data['means'][data_type], data['std_devs'][data_type])):
            if means:  # Vérifie si des données sont disponibles
                label = f'{data_type.capitalize()}_{keywords[i]}'
                x_values = indices[:len(means)]  # Tronquer les indices si nécessaire
                plt.plot(x_values, means, linestyle='-', label=f'{label}_Moyenne')
                plt.fill_between(x_values, [m + s for m, s in zip(means, std_devs)],
                                 [m - s for m, s in zip(means, std_devs)], alpha=0.2, linestyle='-',
                                 label=f'{label}_Écart_Type')

        # Ajoutez les courbes correspondant aux valeurs maximales et minimales
        if data_type == 'valid':
            for i, (max_values, min_values) in enumerate(zip(max_list['valid'], min_list['valid'])):
                label = f'{keywords[i]}'
                x_values = indices[:len(max_values)]
                plt.plot(x_values, max_values, linestyle='--', label=f'{label}_Max', color=colors[i])
                plt.plot(x_values, min_values, linestyle='--', label=f'{label}_Min', color=colors[i])

    for i, (mean_localSearch, std_dev_localSearch) in enumerate(localSearch_data):
        # Utilisez la couleur correspondante à chaque mot-clé
        color = colors[i]
        plt.axhline(y=mean_localSearch, color='red', linestyle='-', label=f'{localSearch_keywords[i]}_Moyenne')
        plt.fill_between(indices, [mean_localSearch + std_dev_localSearch] * len(indices),
                         [mean_localSearch - std_dev_localSearch] * len(indices), color='red', alpha=0.2,
                         label=f'{localSearch_keywords[i]}_Écart_Type')

    plt.xlabel('Number of Generations')
    plt.ylabel('Mean / E-Type')
    plt.title('Evolution of Mean and Std Dev vs. Number of Generations')
    plt.legend()
    plt.grid(True)

    if save_filename:
        # Génère un nom de fichier incremental pour le graphique
        num = 1
        filename = f'evolution_graph_{num}.png'
        while os.path.isfile(os.path.join("./Res", filename)):
            num += 1
            filename = f'evolution_graph_{num}.png'

        # Enregistre le graphique dans un fichier image incremental
        save_plot_to_res(filename)

    plt.show()

if __name__ == "__main__":
    directory = './results_09102023'
    keywords = ['strategyNN_10,5_N_128_K_8']
    localSearch_keywords = ['hillClimber_10,5_N_128_K_8']

    # Mettez à jour la manière dont vous déballez les valeurs renvoyées par extract_and_process_data
    data_matrices, means_list, std_devs_list, max_list, min_list = extract_and_process_data(directory, keywords)

    if means_list['train']:
        indices = list(range(1, len(means_list['train'][0]) + 1))
    else:
        indices = []  # Ou définissez des indices par défaut

    # Extraction des données localSearch
    localSearch_data = []
    for local_keyword in localSearch_keywords:
        local_search_data = extract_localSearch_data(directory, [local_keyword])
        localSearch_data.append(local_search_data)

    # Passage des données localSearch extraites à la fonction de traçage
    plot_data_evolution(indices, {'means': means_list, 'std_devs': std_devs_list}, localSearch_data,
                        max_list, min_list, save_filename="mon_graphique.png")
