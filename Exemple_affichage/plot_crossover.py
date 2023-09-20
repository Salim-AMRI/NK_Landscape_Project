import pandas as pd
import numpy as np

from plotly.subplots import make_subplots

import plotly.graph_objs as go
from os import walk




instances = ["G58","G61","G64","G70"]



crossovers = ["XT", "UX", "MX", "PR"]
#crossovers = ["UX", "PR", "XT"]

list_color = [(0,204,0),(204,0,0),(0,0,204), (204,204,0),(0,204,204)]

list_bounds = [(19270,19295),(5780,5800), (8720,8760),(9580,9600)]

list_pos_subplot = [(1,1),(1,2), (2,1),(2,2)]


list_begin_size = [15, 15, 15, 15]
#list_begin_size = [0, 0, 0, 0]

list_end_size = [1000, 1000, 1000, 1000]

mypath = "./results_crossovers/"


fichier = open("results_crossovers.csv", "w")

fichier.write( "Instance,Crossover, Min score,Avg score,Best time" + "\n" )


f = []
for (dirpath, dirnames, filenames) in walk(mypath):

    f = filenames
    break;

min_time = -1




fig = make_subplots(rows=2, cols=2, subplot_titles=("G58","G61","G64","G70"))


for i, instance in enumerate(instances):

    for idx, crossover in enumerate(crossovers):

        min_scores = []
        min_times = []


        min_nb_gen = 999999
        
        for file in filenames:

            if (instance in file and crossover in  file):

                data = np.loadtxt(mypath + file, delimiter=",")    
            
                nb_gen = data.shape[0]
                
                if(nb_gen < min_nb_gen):
                    
                    min_nb_gen = nb_gen
        
        matrix_score = np.zeros((min_nb_gen, 10))
        
        cpt = 0
        
        for file in filenames:

            if (instance in file and crossover in  file):

                # print(instance)
                data = np.loadtxt(mypath + file, delimiter=",")                


                matrix_score[:,cpt] = -data[:min_nb_gen, 0]

                
                cpt += 1


                min_score = np.min(data[:,0])

                #print(min_score)

                for l in range(data.shape[0]):
                    if(data[l,0] == min_score):
                        min_time = data[l,3]

                        break;

                min_scores.append(min_score)
                min_times.append(min_time)




        x = list(np.array(range(min_nb_gen)))[list_begin_size[i]:list_end_size[i]]
        y = list(np.mean(matrix_score,1))[list_begin_size[i]:list_end_size[i]]
        y_lower = list(np.mean(matrix_score,1) - np.std(matrix_score,1) )[list_begin_size[i]:list_end_size[i]]
        y_upper = list(np.mean(matrix_score,1) + np.std(matrix_score,1))[list_begin_size[i]:list_end_size[i]]

        color = list_color[idx]

        if(crossover == "XT"):
            crossover = "RTSC"

        if(i == 0):
            fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    name=crossover,
                    line=dict(color='rgb(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ')'),
                    mode='lines'
                ),
                row=list_pos_subplot[i][0], col=list_pos_subplot[i][1]
            )
        else:
            fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    name=crossover,
                    line=dict(color='rgb(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ')'),
                    mode='lines',
                    showlegend=False
                ),
                row=list_pos_subplot[i][0], col=list_pos_subplot[i][1]

            )


        fig.add_trace(go.Scatter(
                x=x+x[::-1], # x, then x reversed
                y=y_upper+y_lower[::-1], # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]) + ',' + str(0.2) + ')',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ),
            row=list_pos_subplot[i][0], col=list_pos_subplot[i][1]
        )

        best_time = []

        best_score = np.min(min_scores)


        for idx, score in enumerate(min_scores):

            if (score == best_score):
                best_time.append(min_times[idx])


        fichier.write(
            instance + "," + crossover + "," + str(-int(best_score)) + "," + str(-np.mean(min_scores)) + "," + str(
                int(np.mean(best_time))) + "\n")



fichier.close()

# fig.update_layout(yaxis_range=[min, max])

fig.update_xaxes(title_text="Number of generations", row=1, col=1, title_font=dict(size=20))
fig.update_xaxes(title_text="Number of generations", row=1, col=2, title_font=dict(size=20))
fig.update_xaxes(title_text="Number of generations", row=2, col=1, title_font=dict(size=20))
fig.update_xaxes(title_text="Number of generations", row=2, col=2, title_font=dict(size=20))

fig.update_yaxes(title_text="Average score", row=1, col=1, title_font=dict(size=20))
fig.update_yaxes(title_text="Average score", row=1, col=2, title_font=dict(size=20))
fig.update_yaxes(title_text="Average score", row=2, col=1, title_font=dict(size=20))
fig.update_yaxes(title_text="Average score", row=2, col=2, title_font=dict(size=20))

fig.update_layout( legend=dict(font=dict(family="Courier", size=30, color="black")),
                  legend_title=dict(font=dict(family="Courier", size=30, color="blue")))

fig.update_annotations(font_size=30)


fig.show()






#print(x)
#fig = go.Figure([

    #go.Scatter(
        #name='AUX',
        #x=x,
        #y=y_aux,
        #mode='lines',
        #line=dict(color='rgb(0, 200, 0)'),
    #),
    #go.Scatter(
        #name='MAGX',
        #x=x,
        #y=y_MAGX,
        #mode='lines',
        #line=dict(color='rgb(200, 0, 0)'),
    #),
    #go.Scatter(
        #name='No Crossover',
        #x=x,
        #y=y_xx,
        #mode='lines',
        #line=dict(color='rgb(0, 0, 200)'),
    #),
    #go.Scatter(
        #name='UX',
        #x=x,
        #y=y_hux,
        #mode='lines',
        #line=dict(color='rgb(0, 200, 200)'),
    #),
    #go.Scatter(
        #name='GPX',
        #x=x,
        #y=y_gpx,
        #mode='lines',
        #line=dict(color='rgb(200, 200, 0)'),
    #),

#])

#fig.update_layout(
    #xaxis_title='Number of generations',
    #yaxis_title='Average score',
    #title='',
    #hovermode="x"
#)
#fig.show()









# for i in range(1,11):
#
#     fileName = "closest/LCS_size_pop_12288_nb_iter_107700_nb_neighbors_1_QC-60-70-1.txt_k_60_2021-12-03 12:13:18.941757.txt"
#
#     data = np.loadtxt(fileName, delimiter=",")



