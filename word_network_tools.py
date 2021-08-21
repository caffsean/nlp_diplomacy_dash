
"""
Created - July 2021
Author - caffsean

This file is the MODULE for making word embedding network graphs for the Vector Space Dashboard

"""


from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import plotly.graph_objs as go
import pandas as pd

def word2vec_network(model, word_list, threshhold=0.5):
    words, vectors = [], []
    for item in word_list:
        try:
            vectors.append(model.wv.get_vector(item))
            words.append(item)
        except:
            print(f'Word {item} not found in vocab.')
    sims = cosine_similarity(vectors, vectors)
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i<=j:
                sims[i, j] = False
    indices = np.argwhere(sims > threshhold)

    G = nx.Graph()

    for index in indices:
        G.add_edge(words[index[0]], words[index[1]], weight=sims[index[0],
                                                                 index[1]])

    weight_values = nx.get_edge_attributes(G,'weight')
    positions = nx.spring_layout(G)
    nx.set_node_attributes(G,name='position',values=positions)
    searches = []
    edge_x = []
    edge_y = []
    weights = []
    ave_x, ave_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['position']
        x1, y1 = G.nodes[edge[1]]['position']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        ave_x.append(np.mean([x0, x1]))
        ave_y.append(np.mean([y0, y1]))
        weights.append(f'{edge[0]}, {edge[1]}: {weight_values[(edge[0], edge[1])]}')
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        opacity=0.3,
        line=dict(width=2, color='White'),
        hoverinfo=None,
        mode='lines')
    edge_trace.text = weights
    node_x = []
    node_y = []
    sizes = []
    for node in G.nodes():
        x, y = G.nodes[node]['position']
        node_x.append(x)
        node_y.append(y)
        if node in searches:
            sizes.append(50)
        else:
            sizes.append(15)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            showscale=False,
            line=dict(color='White'),
            colorscale='RdBu',
            reversescale=False,
            color=[],
            opacity=0.9,
            size=sizes,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )
    invisible_similarity_trace = go.Scatter(
        x=ave_x, y=ave_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=[],
            opacity=0,
        )
    )
    invisible_similarity_trace.text=weights

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(adjacencies[0])
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    fig = go.Figure(
        data=[edge_trace, node_trace, invisible_similarity_trace],
        layout=go.Layout(
            title=None,
            template='plotly_dark',
            titlefont_size=20,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=20,r=20,t=40),
            annotations=[
                dict(
                    text='',
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    fig.update_traces(showlegend=False)
    return fig


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

import scipy as sp
countries = ['afghanistan', 'albania', 'algeria', 'angola', 'argentina', 'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bolivia', 'bosnia', 'botswana', 'brazil', 'brunei', 'bulgaria', 'burkina', 'burundi', 'cabo', 'cambodia', 'cameroon', 'canada', 'chad', 'chile', 'china', 'colombia', 'comoros', 'congo', 'costa', 'côte', 'croatia', 'cuba', 'cyprus', 'czechia', 'denmark', 'djibouti', 'dominican', 'ecuador', 'egypt', 'elsalvador', 'equatorialguinea', 'eritrea', 'estonia', 'eswatini', 'ethiopia', 'fiji', 'finland', 'france', 'gabon', 'georgia', 'germany', 'ghana', 'greece', 'guatemala', 'guineabissau', 'guinea', 'guyana', 'haiti','vatican','honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq', 'ireland', 'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya', 'northkorea','southkorea', 'kuwait', 'kyrgyzstan', 'lao', 'latvia', 'lebanon', 'lesotho', 'liberia', 'libya', 'lithuania', 'luxembourg', 'madagascar', 'malawi', 'malaysia', 'mali', 'malta', 'mauritania', 'mauritius', 'mexico', 'moldova', 'mongolia', 'montenegro', 'morocco', 'mozambique', 'myanmar', 'nepal', 'netherlands', 'newzealand', 'nicaragua', 'niger', 'nigeria', 'norway', 'oman', 'pakistan', 'palau', 'palestine', 'panama', 'paraguay', 'peru', 'philippines', 'poland', 'portugal', 'qatar', 'romania', 'russia', 'rwanda', 'saudi', 'saudiarabia','senegal', 'serbia', 'seychelles', 'sierra', 'singapore', 'slovakia', 'slovenia', 'somalia', 'spain', 'srilanka', 'sudan', 'suriname', 'sweden', 'switzerland', 'syria', 'tajikistan', 'tanzania', 'thailand', 'togo', 'trinidad', 'tunisia', 'turkey', 'turkmenistan', 'uganda', 'ukraine', 'uae', 'emirates', 'unitedstates', 'uruguay', 'uzbekistan', 'venezuela', 'vietnam', 'yemen', 'zambia', 'zimbabwe']

def get_mahalanobis_outlier_rank(model, category_list):
    matrix = []
    for item in category_list:
        vector = model.wv.get_vector(item)
        matrix.append(vector)
    matrix = np.vstack(matrix)
    vector_df = pd.DataFrame(matrix)
    vector_df.index = category_list
    columns = list(range(100))
    df_x = vector_df[columns]
    vector_df['mahalanobis'] = mahalanobis(x=df_x, data=vector_df[columns])
    return pd.DataFrame(vector_df['mahalanobis'].sort_values(ascending=False))
