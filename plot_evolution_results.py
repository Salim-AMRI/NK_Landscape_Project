import pandas as pd
import numpy as np

from plotly.subplots import make_subplots

import plotly.graph_objs as go
from os import walk
from strategies.StrategyNNFitness_and_current import StrategyNNFitness_and_current



N = 64
K = 8


list_LS = [ "strategyNN_", "StrategyNNFitness_and_current", "strategyNNRanked_v2_10", "strategyNNRanked_v2_zScore","hillClimberJump"]



list_color = [(0,204,0),(204,0,0),(0,0,204), (204,204,0),(0,204,204)]




mypath = "./results/"



f = []
for (dirpath, dirnames, filenames) in walk(mypath):

    f = filenames
    break;

min_time = -1


fig = go.Figure()

for idx, LS in enumerate(list_LS):

    min_scores = []
    min_times = []

    
    if("NN" in LS):
        
        min_nb_gen = 999999
        cpt = 0

        for file in filenames:

            if ( LS in  file and ("_" + str(N) + "_") in  file and ("K_" + str(K)) in  file):

                data = pd.read_csv(mypath + file, delimiter=",").values    
            
                nb_gen = data.shape[0]

                cpt += 1
                if(nb_gen < min_nb_gen):
                    
                    min_nb_gen = nb_gen

        min_nb_gen = 100

        print("cpt : " + str(cpt))
        matrix_score = np.zeros((min_nb_gen, cpt))
    
    else:
        matrix_score = np.zeros((1, 10))
        
    cpt = 0

    min_nb_gen = 100

    for file in filenames:

        if ( LS in  file and ("_" + str(N) + "_") in  file and ("K_" + str(K)) in  file):

            print(file)
            data = pd.read_csv(mypath + file, delimiter=",").values              


            if("NN" in LS):
                matrix_score[:,cpt] = data[:min_nb_gen, 2]
                
            else:
                matrix_score[0,cpt] = data[0,2]

            
            cpt += 1

    matrix_score = matrix_score/N

    print(min_nb_gen)


    x = list(np.array(range(min_nb_gen)))
    
    if("NN" in LS):
    
        y = list(np.mean(matrix_score,1))
        y_lower = list(np.min(matrix_score,1) )
        y_upper = list(np.max(matrix_score,1) )

    else:
        mean = np.mean(matrix_score,1)[0]
        min = np.min(matrix_score,1)[0]
        max = np.max(matrix_score,1)[0]

        y = [mean for i in range(min_nb_gen)]
        y_lower = [min for i in range(min_nb_gen)]
        y_upper = [max for i in range(min_nb_gen)]

        print(y)
        print(y_lower)
        print(y_upper)
        

    color = list_color[idx]

    #"strategyNN_", "StrategyNNFitness_and_current", "strategyNNRanked_v2_10", "strategyNNRanked_v2_zScore", "hillClimberJump"]



    nameAlgo = LS
    if(LS == "strategyNN_"):
        nameAlgo = "o1"
    if(LS == "StrategyNNFitness_and_current"):
        nameAlgo = "o2 "

    if(LS == "strategyNNRanked_v2_10"):
        nameAlgo = "o3"
    if(LS == "strategyNNRanked_v2_zScore"):
        nameAlgo = "o4"

    if ("NN" in LS):
        fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    name=nameAlgo,
                    line=dict(color='rgb(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ')'),
                    mode='lines',
                    legendgroup="NeuroLS",
                    legendgrouptitle={'text': 'Neuro-LS'}
                )
            )
    else:
        fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    name="BHC+",
                    line=dict(color='rgb(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ')'),
                    mode='lines',
                    legendgroup="Baseline",
                    legendgrouptitle={'text': 'Baseline'}

                )
            )

    fig.add_trace(go.Scatter(
            x=x+x[::-1], # x, then x reversed
            y=y_upper+y_lower[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ',' + str(0.2) + ')',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,

        )
    )






# fig.update_layout(yaxis_range=[min, max])

fig.update_xaxes(title_text="Number of CMA-ES generations", title_font=dict(size=12))


fig.update_yaxes(title_text="Average validation score", title_font=dict(size=12))

#fig.update_layout( legend=dict(font=dict(family="Courier", size=20, color="black")),
#                  legend_title=dict(font=dict(family="Courier", size=20, color="blue")))

fig.update_annotations(font_size=30)


fig.show()
#fig.write_image("images/results_neuroevolution_N_" + str(N) + "_K_" + str(K) + ".png")


