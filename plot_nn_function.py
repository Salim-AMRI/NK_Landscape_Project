import numpy as np
from joblib import Parallel, delayed
import argparse
from Main import get_Score_trajectory
import json

import os

import plotly

import plotly.graph_objs as go


#from scipy import stats
#from scipy.stats import shapiro

#parser.add_argument('--nb_restarts', type=int, default=1, help='Nombre de redémarrages')
#parser.add_argument('--nb_instances', type=int, default=10, help='Nombre d\'instances')
#parser.add_argument('--nb_jobs', type=int, default=10, help='Nombre de jobs')

#args = parser.parse_args()

# Paramètres initiaux en fonction des argumentssolution = np.loadtxt(solution_file)


# Chemin du répertoire contenant les fichiers à consulter
#directory = 'results_09102023/'


log_directory = 'log_trajectory'

input_list = []
input_list_bis = []
output_list = []

results_dict = {}


fig = go.Figure()

#list_strategy_name = [ "strategyNN", "StrategyNNFitness_and_current", "StrategyNNRanked_v2", "strategyNNRanked_v2_zScore"]


list_strategy_name = [ "StrategyNNFitness_and_current"]
#list_strategy_name = [ "strategyNN"]

for N in [64]:
    for K in [ 8]:

        dico_results_strategy = {}

        for type_strategy in list_strategy_name:

            for i in range(10):

                filename = "input_" + type_strategy + "_N_64_K_8_nb_instances_test_" + str(i) + "_nb_restarts_0.log"

                file_path = log_directory + "/" + filename
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    for line in lines:
                        # Sépare la ligne en éléments en utilisant la virgule comme séparateur
                        res = json.loads(line)

                        print(res)

                        if(type_strategy == "StrategyNNRanked_v2_zScore" or type_strategy == "StrategyNNFitness_and_current"):
                            input_list.extend(res[0])
                            input_list_bis.extend(res[1])

                            print("resr0")
                            print(res[0])
                            print("resr1")
                            print(res[1])
                        else:
                            input_list.extend(res)


                        #print(res)

            for i in range(10):

                filename = "output_" + type_strategy + "_N_64_K_8_nb_instances_test_" + str(i) + "_nb_restarts_0.log"
                file_path = log_directory + "/" + filename
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    for line in lines:
                        # Sépare la ligne en éléments en utilisant la virgule comme séparateur

                        res = json.loads(line)

                        output_list.extend(res)


            #print("output_list")
            #print(len(output_list))


#

fig = go.Figure(data=[go.Scatter3d(x=input_list, y=input_list_bis, z=output_list,mode='markers',
                                   marker=dict(
                                       size=12,
                                       color=output_list,  # set color to an array/list of desired values
                                       colorscale=["red", "green"],  # choose a colorscale
                                       opacity=0.8
                                   ))] )

'''
fig.update_layout(
    autosize=False,

    scene=dict(
        xaxis_title="Ranking score",
        yaxis_title="Z-score",
        zaxis_title="Preference score",

    ),
)
'''

fig.update_layout(
    autosize=False,

    scene=dict(
        xaxis_title="Current fitness",
        yaxis_title="New fitness",
        zaxis_title="Preference score",

    ),
)

fig.show()

#fig.update_xaxes(title_text="Ranking score", title_font=dict(size=12))
#fig.update_yaxes(title_text="Z-score", title_font=dict(size=12))
#fig.update_yaxes(title_text="Preference score of the move", title_font=dict(size=12))
#fig.update_annotations(font_size=30)




#fig = go.Figure(data=[go.Scatter2d(x=input_list, y=input_list_bis,
#                                   mode='markers')])

#fig.show()



'''
import plotly.express as px
fig = px.scatter(x=input_list, y=output_list, color=output_list, color_continuous_scale=["red", "green"])

fig.update_xaxes(title_text="Variation of fitness with the move (Delta)", title_font=dict(size=16))
#fig.update_xaxes(title_text="Ranking score", title_font=dict(size=16))
fig.update_yaxes(title_text="Preference score", title_font=dict(size=16))

fig.show()
'''