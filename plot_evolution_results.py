import pandas as pd
import numpy as np

from plotly.subplots import make_subplots

import plotly.graph_objs as go
from os import walk



N = 32
K = 2


list_LS = [ "NN_withTabu","NN_3","_hillClimber", "tabu" ,"IteratedhillClimber"]



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


   
    
    if(LS == "NN_3" or LS == "NN_withTabu"):
        
        min_nb_gen = 999999
        
        for file in filenames:

            if ( LS in  file and ("_" + str(N) + "_") in  file and ("K_" + str(K)) in  file):

                data = pd.read_csv(mypath + file, delimiter=",").values    
            
                nb_gen = data.shape[0]
                
                print(data)
                
                if(nb_gen < min_nb_gen):
                    
                    min_nb_gen = nb_gen
    
        matrix_score = np.zeros((min_nb_gen, 10))
    
    else:
        matrix_score = np.zeros((1, 10))
        
    cpt = 0
    
    for file in filenames:

        if ( LS in  file and ("_" + str(N) + "_") in  file and ("K_" + str(K)) in  file):

            print(file)
            data = pd.read_csv(mypath + file, delimiter=",").values              


            if(LS == "NN_3" or LS == "NN_withTabu"):
                matrix_score[:,cpt] = data[:min_nb_gen, 2]
                
            else:
                matrix_score[0,cpt] = data[0,2]

            
            cpt += 1



    print(min_nb_gen)


    x = list(np.array(range(min_nb_gen)))
    
    if(LS == "NN_3" or LS == "NN_withTabu"):
    
        y = list(np.mean(matrix_score,1))
        y_lower = list(np.mean(matrix_score,1) - np.std(matrix_score,1) )
        y_upper = list(np.mean(matrix_score,1) + np.std(matrix_score,1))

    else:
        mean = np.mean(matrix_score,1)[0]
        std = np.std(matrix_score,1)[0]
        
        y = [mean for i in range(min_nb_gen)]
        y_lower = [(mean - std) for i in range(min_nb_gen)]
        y_upper = [(mean + std) for i in range(min_nb_gen)]

        print(y)
        print(y_lower)
        print(y_upper)
        

    color = list_color[idx]

    nameAlgo = LS
    if(LS == "NN_3"):
        nameAlgo = "NN"
    if(LS == "_hillClimber"):
        nameAlgo = "hillClimber"
    if(LS == "IteratedhillClimber"):
        nameAlgo = "Iterated hillClimber"    
    
    fig.add_trace(go.Scatter(
                x=x,
                y=y,
                name=nameAlgo,
                line=dict(color='rgb(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ')'),
                mode='lines'
            )
        )
    


    fig.add_trace(go.Scatter(
            x=x+x[::-1], # x, then x reversed
            y=y_upper+y_lower[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ',' + str(0.2) + ')',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    )




# fig.update_layout(yaxis_range=[min, max])

fig.update_xaxes(title_text="Number of generations", title_font=dict(size=20))


fig.update_yaxes(title_text="Average score", title_font=dict(size=20))

fig.update_layout(title=dict(text="N_" + str(N) + "_K_" + str(K), font=dict(size=40)), legend=dict(font=dict(family="Courier", size=20, color="black")),
                  legend_title=dict(font=dict(family="Courier", size=20, color="blue")))

fig.update_annotations(font_size=30)



fig.write_image("images/results_neuroevolution_N_" + str(N) + "_K_" + str(K) + ".png")


