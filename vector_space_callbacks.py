
"""
Created - July 2021
Author - caffsean
"""

import plotly.graph_objects as go
from dash.dependencies import Output, Input, State, ALL, ALLSMALLER, MATCH
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

from app import app
from my_dictionaries import language_dictionary,label_dictionary
import word_network_tools

tfidf_pca_df = pd.read_csv('assets/tfidf_pca_df.csv')
bert_pca_df = pd.read_csv('assets/bert_pca_df.csv')
bert_df = pd.read_csv('assets/BERT_sentence_averages.csv')
bert_outliers_rus_df = pd.read_csv('assets/BERT_outliers_rus.csv')
bert_outliers_usa_df = pd.read_csv('assets/BERT_outliers_usa.csv')
bert_divergence = pd.read_csv('assets/BERT_divergence.csv')
us_model = Word2Vec.load("assets/language_models/us_en_w2v.model")
rus_model = Word2Vec.load("assets/language_models/rus_en_w2v.model")

countries = ['afghanistan', 'albania', 'algeria', 'angola', 'argentina', 'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bolivia', 'bosnia', 'botswana', 'brazil', 'brunei', 'bulgaria', 'burkina', 'burundi', 'cabo', 'cambodia', 'cameroon', 'canada', 'chad', 'chile', 'china', 'colombia', 'comoros', 'congo', 'costa', 'côte', 'croatia', 'cuba', 'cyprus', 'czechia', 'denmark', 'djibouti', 'dominican', 'ecuador', 'egypt', 'elsalvador', 'equatorialguinea', 'eritrea', 'estonia', 'eswatini', 'ethiopia', 'fiji', 'finland', 'france', 'gabon', 'georgia', 'germany', 'ghana', 'greece', 'guatemala', 'guineabissau', 'guinea', 'guyana', 'haiti','vatican','honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq', 'ireland', 'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya', 'northkorea','southkorea', 'kuwait', 'kyrgyzstan', 'lao', 'latvia', 'lebanon', 'lesotho', 'liberia', 'libya', 'lithuania', 'luxembourg', 'madagascar', 'malawi', 'malaysia', 'mali', 'malta', 'mauritania', 'mauritius', 'mexico', 'moldova', 'mongolia', 'montenegro', 'morocco', 'mozambique', 'myanmar', 'nepal', 'netherlands', 'newzealand', 'nicaragua', 'niger', 'nigeria', 'norway', 'oman', 'pakistan', 'palau', 'palestine', 'panama', 'paraguay', 'peru', 'philippines', 'poland', 'portugal', 'qatar', 'romania', 'russia', 'rwanda', 'saudi', 'saudiarabia','senegal', 'serbia', 'seychelles', 'sierra', 'singapore', 'slovakia', 'slovenia', 'somalia', 'spain', 'srilanka', 'sudan', 'suriname', 'sweden', 'switzerland', 'syria', 'tajikistan', 'tanzania', 'thailand', 'togo', 'trinidad', 'tunisia', 'turkey', 'turkmenistan', 'uganda', 'ukraine', 'uae', 'emirates', 'unitedstates', 'uruguay', 'uzbekistan', 'venezuela', 'vietnam', 'yemen', 'zambia', 'zimbabwe']

us_score_df = word_network_tools.get_mahalanobis_outlier_rank(us_model, countries).reset_index()
rus_score_df = word_network_tools.get_mahalanobis_outlier_rank(rus_model, countries).reset_index()

bgcolor1='#5D6D7E'
bgcolor2='#85929E'
template = 'plotly_dark'


def word2vec_word_association_network(word, model):
    word = word.lower()
    word_associations = [word]+[word[0] for word in model.wv.most_similar([word],topn=60)]
    return word_network_tools.word2vec_network(model, word_associations, threshhold=0.7)

def word2vec_word_association_with_countries(model):
    countries = ['usa','afghanistan', 'albania', 'algeria', 'angola', 'argentina', 'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bolivia', 'bosnia', 'botswana', 'brazil', 'brunei', 'bulgaria', 'burkina', 'burundi', 'cabo', 'cambodia', 'cameroon', 'canada', 'chad', 'chile', 'china', 'colombia', 'comoros', 'congo', 'costa', 'côte', 'croatia', 'cuba', 'cyprus', 'czechia', 'denmark', 'djibouti', 'dominican', 'ecuador', 'egypt', 'elsalvador', 'equatorialguinea', 'eritrea', 'estonia', 'eswatini', 'ethiopia', 'fiji', 'finland', 'france', 'gabon', 'georgia', 'germany', 'ghana', 'greece', 'guatemala', 'guineabissau', 'guinea', 'guyana', 'haiti', 'holysee','vatican','honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq', 'ireland', 'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya', 'northkorea','southkorea', 'kuwait', 'kyrgyzstan', 'lao', 'latvia', 'lebanon', 'lesotho', 'liberia', 'libya', 'lithuania', 'luxembourg', 'madagascar', 'malawi', 'malaysia', 'mali', 'malta', 'mauritania', 'mauritius', 'mexico', 'micronesia', 'moldova', 'mongolia', 'montenegro', 'morocco', 'mozambique', 'myanmar', 'nepal', 'netherlands', 'newzealand', 'nicaragua', 'niger', 'nigeria', 'norway', 'oman', 'pakistan', 'palau', 'palestine', 'panama', 'paraguay', 'peru', 'philippines', 'poland', 'portugal', 'qatar', 'romania', 'russia', 'rwanda', 'samoa', 'saudi', 'saudiarabia','senegal', 'serbia', 'seychelles', 'sierraleone', 'singapore', 'slovakia', 'slovenia', 'somalia', 'spain', 'srilanka', 'sudan', 'suriname', 'sweden', 'switzerland', 'syria', 'tajikistan', 'tanzania', 'thailand', 'timor', 'togo', 'trinidad', 'tunisia', 'turkey', 'turkmenistan', 'uganda', 'ukraine', 'uae', 'unitedstates', 'uruguay', 'uzbekistan', 'venezuela','vietnam','yemen', 'zambia', 'zimbabwe'] ## removed unitedarabemirates, america, viet, papua
    return word_network_tools.word2vec_network(model, countries, threshhold=0.3)

@app.callback(
            Output('graph-vector-space','figure'),
            Output('pca-header','children'),
            [Input('color-switch','on'),
            Input('bert-switch','on'),
            Input('cluster-slider','value'),
            Input('dim-switch','on'),
            Input('k-switch','on'),
            Input('text-switch','on')])
def update_clusters(color,data,n_clusters,dim,show_k,text):
    if data == False:
        dff = tfidf_pca_df
    else:
        dff = bert_pca_df

    if text == False:
        marker_text = 'markers'
    else:
        marker_text = 'markers+text'

    if color == False:
        color_variable = dff['binary_class']
    else:
        color_variable = dff['continent_colors']

    if dim == False:
        dimension = '2D'
    else:
        dimension = '3D'

    if dimension == '2D':
        trace_1 = go.Scatter(
            x=dff['2D-0'],
            y=dff['2D-1'],
            text=dff['Country'],
            textposition = 'top center',
            mode=marker_text,
            marker=dict(
                size=16,
                color=color_variable,                # set color to an array/list of desired values
                colorscale='spectral',   # choose a colorscale
                opacity=0.6,
                line=dict(
                        color='MediumPurple',
                        width=1)),
            showlegend=False)
        if show_k == True:
            km = KMeans(n_clusters)
            clusts = km.fit_predict(np.array(dff[['2D-0','2D-1']]))
            centers = pd.DataFrame(km.cluster_centers_)#.transpose()
            trace_2 = go.Scatter(
                x = centers[0],
                y = centers[1],
                mode='markers+text',
                marker=dict(
                    size=123,
                    color='orange',
                    opacity=0.5),
                showlegend=False
            )
            fig = go.Figure(data=[trace_2,trace_1])
        else:
            fig = go.Figure(data=[trace_1])
    elif dimension == '3D':
        trace_1 = go.Scatter3d(
            x=dff['3D-0'],
            y=dff['3D-1'],
            z=dff['3D-2'],
            text=dff['Country'],
            textposition = 'top center',
            mode=marker_text,
            marker=dict(
                size=6,
                color=color_variable,                # set color to an array/list of desired values
                colorscale='spectral',   # choose a colorscale
                opacity=0.6,
                line=dict(
                        color='MediumPurple',
                        width=1)),
            showlegend=False)
        if show_k == True:
            km = KMeans(n_clusters)
            clusts = km.fit_predict(np.array(dff[['3D-0','3D-1','3D-2']]))
            centers = pd.DataFrame(km.cluster_centers_)#.transpose()
            trace_2 = go.Scatter3d(
                x = centers[0],
                y = centers[1],
                z = centers[2],
                mode='markers+text',
                marker=dict(
                    size=123,
                    color='orange',
                    opacity=0.5),
                showlegend=False
            )
            fig = go.Figure(data=[trace_2,trace_1])
        else:
            fig = go.Figure(data=[trace_1])

    fig.update_traces(textposition='top center')
    fig.update_layout(height=800)
    camera = dict(eye=dict(x=2, y=2, z=0.1),
                center=dict(x=0.5, y=0.7, z=0),
                up=dict(x=0, y=0, z=1))

    fig.update_layout(scene_camera=camera)
    fig.update_traces(textposition='top center')
    fig.update_layout(plot_bgcolor=bgcolor1)
    fig.update_layout(scene=dict(aspectratio=dict(x=2.5,y=2.5,z=2.5)))
    fig.update_layout(scene=dict(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightPink'),
                                     yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightPink'),
                                     zaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightPink'),
                                ),
                         )
    fig.update_layout(template=template)


    if data == False:
        title = 'Explore: PCA of TFIDF Vector Space'
    else:
        title = 'Explore: PCA of BERT Vector Space'
    return fig, title



## Update Embassy Choice Dropdown - (Prevents Countries that don't have data from appearing in dropdown)
@app.callback(
            Output('dropdown-embassy-BERT','options'),
            Output('dropdown-embassy-BERT','value'),
            [Input('dropdown-network-BERT','value')])
def update_country_dropdowns(network):
    if network == 'Russia':
        options_list = list(bert_df[bert_df.network=='RUS'].country)
    else:
        options_list = list(bert_df[bert_df.network=='USA'].country)
    options = [{'label': i, 'value': i} for i in options_list]
    value = options[0]['label']
    return options, value

## Graph - Bert Similarity
@app.callback(Output('bert-similarity','figure'),
            Output('bert-similarity-header','children'),
            Input('dropdown-embassy-BERT','value'),
            Input('dropdown-network-BERT','value'))
def update_BERT_similarities(country,network):
    if network == 'Russia':
        base = 'RUS'
    else:
        base = 'USA'
    country_array = np.array(bert_df[(bert_df.country==country)&(bert_df.network==base)][list(bert_df.columns)[1:-2]].iloc[0]).reshape(1,-1)
    X = bert_df[list(bert_df.columns)[1:-2]]
    sims  = []
    for idx in range(len(X)):
        sim = cosine_similarity(country_array, np.array(X.iloc[idx]).reshape(1,-1))
        sims.append(np.round(sim[0][0],6))
    top_df = pd.DataFrame(sims,columns=['cosine_similarity'])
    top_df['country'] = bert_df['country']
    top_df['network'] = bert_df['network']
    top_df = top_df.sort_values(by='cosine_similarity',ascending=False)
    top_df = top_df[['country','network','cosine_similarity']]

    fig = go.Figure(data=[go.Table(header=dict(values=['Most Similar Accounts','Network','Cosine Similarity'],
                                    line_color='darkslategray',
                                    align='center',
                                    font=dict(color='black', family="Lato", size=20),
                                    height=30
                                    ),
                  columnorder = [1,2,3],
                  columnwidth = [60,30,30],
                  cells=dict(values=[list(top_df['country']), list(top_df['network']), list(top_df['cosine_similarity'])],
                                 fill_color='grey',
                                 line_color='darkslategray',
                                 align='left',
                                 font=dict(color='black', family="Lato", size=20),
                                 height=30)
                                 )])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(template=template)

    title = f'Accounts Most Similar to {network} in {country}'
    return fig, title

## Graph - Bert Outliers
@app.callback(Output('bert-outliers','figure'),
            Output('bert-outliers-header','children'),
            Input('dropdown-network-BERT','value'))
def update_BERT_outliers(network):
    if network == 'United States of America':
        df = bert_outliers_usa_df
    elif network == 'Russia':
        df = bert_outliers_rus_df
    df['mahalanobis'] = np.round(df['mahalanobis'],4)
    fig = go.Figure(data=[go.Table(header=dict(values=['Country of Interest', 'Mahalanobis Score'],
                                    line_color='darkslategray',
                                    align='center',
                                    font=dict(color='black', family="Lato", size=20),
                                    height=30
                                    ),
                  columnorder = [1,2],
                  columnwidth = [60,30],
                  cells=dict(values=[list(df['country']), list(df['mahalanobis'])],
                                 fill_color='grey',
                                 line_color='darkslategray',
                                 align='left',
                                 font=dict(color='black', family="Lato", size=20),
                                 height=30)
                                 )])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(template=template)

    title = f'Outlier Accounts of {network}'
    return fig, title

## Graph - Bert Divergence
@app.callback(Output('bert-divergence','figure'),
            Input('dropdown-network-BERT','value'))
def update_BERT_divergence(network):
    df = bert_divergence
    df['cosine_similarity'] = np.round(df['cosine_similarity'],4)
    fig = go.Figure(data=[go.Table(header=dict(values=['Country of Interest', 'Counterpart Cosine Score'],
                                    line_color='darkslategray',
                                    align='center',
                                    font=dict(color='black', family="Lato", size=20),
                                    height=30
                                    ),
                  columnorder = [1,2],
                  columnwidth = [60,30],
                  cells=dict(values=[list(df['country']), list(df['cosine_similarity'])],
                                 fill_color='grey',
                                 line_color='darkslategray',
                                 align='left',
                                 font=dict(color='black', family="Lato", size=20),
                                 height=30)
                                 )])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(template=template)

    return fig

## Graph - Word Embeddings - Country Names
@app.callback(
            Output('graph-country-vectors','figure'),
            Output('header_embassy_vector_space','children'),
            [Input('dropdown-network','value')])
def update_network(network):
    if network == 'United States of America':
        model = us_model
    elif network == 'Russia':
        model = rus_model
    fig = word2vec_word_association_with_countries(model)
    fig.update_layout(height=600)
    return fig, f'Word Embeddings Network of {network}'

## Table - Mahanalobis Outlier Table
@app.callback(
            Output('mahanalobis-table','figure'),
            [Input('dropdown-network','value')])
def update_mahalanobis(network):
    if network == 'United States of America':
        df = us_score_df
    elif network == 'Russia':
        df = rus_score_df
    df['mahalanobis'] = np.round(df['mahalanobis'],4)
    fig = go.Figure(data=[go.Table(header=dict(values=['Country of Interest', 'Mahalanobis Score'],
                                    line_color='darkslategray',
                                    align='center',
                                    font=dict(color='black', family="Lato", size=20),
                                    height=30
                                    ),
                  columnorder = [1,2],
                  columnwidth = [60,30],
                  cells=dict(values=[list(df['index']), list(df['mahalanobis'])],
                                 fill_color='grey',
                                 line_color='darkslategray',
                                 align='left',
                                 font=dict(color='black', family="Lato", size=20),
                                 height=30)
                                 )])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(template=template)
    fig.update_layout(height=600)
    return fig

## Graph - Searched Embeddings
@app.callback(
            Output('graph-explore-vectors','figure'),
            Output('header_search_vector_space','children'),
            [Input('dropdown-network-search','value'),
            Input('search','value')])
def update_search(network, word):
    if network == 'United States of America':
        model = us_model
    elif network == 'Russia':
        model = rus_model
    fig = word2vec_word_association_network(word, model)
    fig.update_layout(height=600)
    return fig, f'Search Word Embeddings Network of {network}'


## World of Popups
@app.callback(
    Output("popover-document-info", "is_open"),
    [Input("popover-document-button", "n_clicks")],
    [State("popover-document-info", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover-document-info-2", "is_open"),
    [Input("popover-document-button-2", "n_clicks")],
    [State("popover-document-info-2", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover-word-info", "is_open"),
    [Input("popover-word-button", "n_clicks")],
    [State("popover-word-info", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover-word-info-2", "is_open"),
    [Input("popover-word-button-2", "n_clicks")],
    [State("popover-word-info-2", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open
