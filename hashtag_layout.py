# -*- coding: utf-8 -*-
"""
Spyder Editor
​
​
This is a temporary script file.
"""
import pandas as pd
import numpy as np
# Import dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
# Import plotly
import plotly.express as px
import sqlalchemy as sa
from sqlalchemy.orm import close_all_sessions
database = 'postgres'
user = 'postgres'
password = 'rKKFiDXpiu6Wbv3'
#host = '192.168.254.42'
host= '47.200.121.209'

from app import app

engine = sa.create_engine(f'postgresql://{user}:{password}@{host}/{user}')
embassy_accounts = pd.read_csv('assets/embassy_accounts.csv')

large_text_white = {'font-family': "Helvetica",'font-weight': '300','font-size': '300%','color': '#F2F2F2'}
large_text_black = {'font-family': "Helvetica",'font-weight': '200','font-size': '300%','color': '#111111'}
medium_text_white = {'font-family': "Helvetica",'font-weight': '110','font-size': '200%','color': '#F2F2F2'}
small_text_white = {'font-family': "Helvetica",'font-weight': '100','font-size': '100%','color': '#F2F2F2'}
dropdown_style = {'color': 'blue','background-color': '#212121'}
graph_background = {'backgroundColor': '#1d1d29','padding': '10px 10px 10px 10px','border':'4px solid', 'border-radius': 10}
graph_background_blue = {'backgroundColor': '#2D455C','padding': '10px 10px 10px 10px','border':'4px solid','border-color':'black','border-radius': 10}
graph_background_black = {'backgroundColor': 'black','padding': '20px 20px 20px 20px','border':'4px solid', 'border-radius': 10}

card_style = {'border-radius':10}

def get_network(state_actor,emb_country,hashtag):
    #get data from the db
    user_id = embassy_accounts[(embassy_accounts.state_actor == state_actor)
                               & (embassy_accounts.country == emb_country)].iloc[0][3]

    tweets_df = pd.read_sql_query(f"SELECT tweets.entities, tweets.entities_labels, tweets.tweet_id  \
                                    FROM tweets  \
                                    LEFT JOIN tweet_hashtags \
                                    ON tweets.tweet_id = tweet_hashtags.tweet_id \
                                    LEFT JOIN hashtags  \
                                    ON tweet_hashtags.hashtag_id = hashtags.hashtag_id \
                                    WHERE (user_id = ('{user_id}') \
                                    AND (tweets.pos IS NOT NULL) \
                                    AND (hashtag = ('{hashtag}'))) " , con=engine)
    #clean up entities
    import re
    df_entities = tweets_df[['entities', 'entities_labels', 'tweet_id']]
    df_entities = df_entities.dropna(subset= ['entities'])
    df_entities['entities']  =df_entities['entities'].apply(lambda x: re.sub("(?<=\d),(?=\d)|(?<=\d), (?=\d)+", " ", x))
    df_entities['entities']  =df_entities['entities'].apply(lambda x: re.sub("\xa0|&| ’s| 's|amp|the", "", x))
    #df_entities['entities']  =df_entities['entities'].apply(lambda x: x.replace("USA", "US"))
    df_entities['entities']  =df_entities['entities'].apply(lambda x: x.strip(" "))
    df_entities.loc[:,('entities')] = df_entities['entities'].apply(lambda x:x.split(","))
    df_entities.loc[:,('entities_labels')] = df_entities['entities_labels'].apply(lambda x:x.split(","))
    #eliminate entries that were not properly converted
    df_entities['entities_len'] = df_entities['entities'].apply(lambda x : len(x))
    df_entities['entities_labels_len'] = df_entities['entities_labels'].apply(lambda x : len(x))
    df_entities['len'] = df_entities['entities_len']  - df_entities['entities_labels_len']
    df_entities = df_entities[df_entities['len']>=0]
    df_entities['len'] = df_entities['entities_labels_len']  - df_entities['entities_len']
    df_entities = df_entities[df_entities['len']>=0]
    #extract edges and nodes
    tweet_ids = df_entities.explode('entities').tweet_id.tolist()
    entities_labels = df_entities.explode('entities_labels').entities_labels.tolist()
    entities = df_entities.explode('entities').entities.tolist()
    df_ents_exploded = pd.DataFrame.from_dict({'entities':entities,
                                        'entities_labels': entities_labels,
                                        'tweet_id': tweet_ids})
    target_entitites = ['GPE', 'PERSON','ORG', 'LOC','NORP']
    df_ents_filtered = df_ents_exploded[df_ents_exploded['entities_labels'].isin(target_entitites)]
    ent_grps = df_ents_filtered.groupby('tweet_id')
    edges =[]
    nodes =[]
    for group in ent_grps.groups.keys():
        df = ent_grps.get_group(group)
        entities = df.entities.tolist()
        entities = [ent for ent in entities if len(ent)>0]
        nodes.extend(entities)
        edges.extend(list(zip(entities, entities[1:])))
    nodes = list(dict.fromkeys(nodes))
    edges = [sorted(x) for x in edges]
    edges = [tuple(x) for x in edges]
    df_edges = pd.DataFrame(pd.DataFrame.from_dict({'edges':edges}).value_counts()).reset_index()
    df_edges['target'] = df_edges['edges'].apply(lambda x: x[0])
    df_edges['source'] = df_edges['edges'].apply(lambda x: x[1])
    df_edges.rename(columns = {0:'weight'}, inplace = True)
    df_edges = df_edges[['source', 'target','weight']]



    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    #layout position

    #pos = nx.spring_layout(G, k=0.2*1/np.sqrt(len(G.nodes())), iterations=20)


    pos = nx.kamada_kawai_layout(G)

    #pos = nx.spectral_layout(G)


    #build elements for cytoscape
    df_nodes  = pd.DataFrame.from_dict(pos).T.reset_index()
    df_nodes.columns = ['node', 'x', 'y']
    df_nodes['deg_centrality'] = df_nodes['node'].map(nx.degree_centrality(G))
    df_nodes['x'] = df_nodes['x']*750
    df_nodes['y'] = df_nodes['y']*750

    nodes =[]
    for index, row in df_nodes.iterrows():
        size = int(np.round(row['deg_centrality']*1000, 0))
        nodes.append({'data':{'id' : row['node'], 'label':row['node'], 'size': size },
                       'position':{'x':row['x'], 'y':row['y']}
                    })

    edges = []

    for index, row in df_edges.iterrows():
        source, target,weight  = row['source'],row['target'],row['weight']
        edges.append({'data':{'source': source, 'target':target, 'weight': weight}, 'classes':'purple', 'size':weight*100})
    network_elements = nodes
    network_elements.extend(edges)

    return network_elements

def get_top_ten(user_id):
            hashtags = pd.read_sql_query(f"SELECT created_at, tweet_hashtags.tweet_id, hashtags.hashtag \
                                    FROM tweets  \
                                    LEFT JOIN tweet_hashtags \
                                    ON tweets.tweet_id =tweet_hashtags.tweet_id \
                                    LEFT JOIN hashtags \
                                    ON tweet_hashtags.hashtag_id=hashtags.hashtag_id \
                                    WHERE user_id = ('{user_id}')",con=engine)
            # find top 10 hashtags the user engaged with and filter the data
            top_10 = list(hashtags.drop_duplicates().value_counts('hashtag')[:10].keys())

            return top_10, hashtags

default_stylesheet = [
    {
        "selector": "node",
        "style": {
            "width": "mapData(size, 0, 100, 20, 60)",
            "height": "mapData(size, 0, 100, 20, 60)",
            "content": "data(label)",
            "font-size": "12px",
            #"text-color":'white',
            "text-valign": "center",
            "text-halign": "center",
            'background-color': '#ff7f0e',
            "border-color": 'white',
            "border-opacity": "1",
            "border-width": "1px",
            "color": "white",
            "font-size": "10%",
            'border-style':'solid'
                    }
    },

        {
        "selector": "edge",
        "style": {
            "width": "data(weight)",

        }}]



hashtag_layout = html.Div([

html.Div([
#dbc.Container([
    html.Div([
            dbc.Row([
                dbc.Col(html.H5('Hashtag Exploration using NLP and Network Analysis',
                             #className = 'text-left mb-4',
                             style = large_text_white),
                             width = 12)]),
            ],style=graph_background),
    html.Div([
            dbc.Row([
                dbc.Col([html.Label('Select State Actor', style =small_text_white, className = 'mb-1'),
                    dcc.Dropdown(
                        id = 'state_actor',
                        options = [{'label': i,
                                    'value': i} for i in
                                   embassy_accounts['state_actor'].unique()],
                        multi = False,
                        value= 'Russia'
                            )], width = 4),
                dbc.Col([html.Label('Select Country', style = small_text_white, className = 'mb-1'),
                    dcc.Dropdown(
                        id = 'emb_country',
                        options = [{'label': i,
                                    'value': i} for i in
                                   embassy_accounts['country'].tolist()],
                        multi = False,
                        value= 'Afghanistan',
                        className = 'mb-3'
                                    )]
                            , width = 4)]),
            dbc.Row([
                dbc.Col([dcc.Graph(id='emb_hashtags',
                                         style  = {'width': '100%', 'height':'500px'},
                                         className = 'text-white mb-4')], width = 12)]),
                ],style=graph_background),
    html.Div([
        dbc.Row([
            dbc.Col(
                html.H5(id='hashtag-header',
                    style=large_text_white,
                     className = 'text-left mb-4'))]),

        dbc.Row([
            dbc.Col(dcc.Dropdown(
                     id = 'hashtag_choice',
                     value='Afghanistan',
                    options = [],
                     multi = False),width = 4
                 )
            ]),
        ],style=graph_background_blue),
    html.Div([
        dbc.Row([
             dbc.Col(
                cyto.Cytoscape(
                id="ner_network",
                autoungrabify = False,
                minZoom = 0.5,
                maxZoom = 3,

                elements=[],
                layout={'name':'preset'},
                style  = {'width': '100%', 'height':'700px'},
                stylesheet=default_stylesheet
            ),width = 12),
                ]),
        ],style=graph_background_black),
],style=graph_background_blue),

html.Div([
    html.Div([
        dbc.Row([
            dbc.Col(
                html.H5("Untagged Entity Discovery with Naive Bayes",
                     #className = 'text-left mb-4',
                     style=large_text_white)

                     ),]),
         dbc.Row([
            dbc.Col(
                html.P("NOTE: A black graph indicates a lack of training data",
                    style=small_text_white,
                     className = 'text-left'))]),


        dbc.Row([
            dbc.Col([dcc.Graph(id='nb_hash',
                                     style  = {'width': '100%', 'height':'500px'},
                                     className = 'text-white mb-4')], width = 12)]),
                ],style=graph_background)
        ],style=graph_background_blue),
]) ## Last Div


@app.callback(
    Output('emb_hashtags', 'figure'),
    [Input('state_actor', 'value'),
     Input('emb_country', 'value')])
def update_top_hashtags(state_actor, emb_country):
        from dash.exceptions import PreventUpdate
        if state_actor is None or emb_country is None:
            raise PreventUpdate
        else:
            user_id = embassy_accounts[(embassy_accounts.state_actor == state_actor)
                                       & (embassy_accounts.country == emb_country)].iloc[0][3]
            top_10, hashtags = get_top_ten(user_id)

            top_hash_tmln =hashtags[hashtags['hashtag'].isin(top_10)]

            # extract relevant time attributes for the groupby object
            from datetime import datetime
            df = top_hash_tmln.copy()
            df['month'] = df['created_at'].apply(lambda x : x.month)
            df['year'] = df['created_at'].apply(lambda x : x.year)
            df['day'] = df['created_at'].apply(lambda x : x.day)

            # calculate montly hashtag use
            t_ht = pd.DataFrame(df.groupby(['hashtag','month','year']).size()).reset_index().sort_values(['year', 'month'])
            t_ht.rename(columns = {0:'count'}, inplace = True)
            t_ht['created_at'] = t_ht['month'].astype(str) + '-'+ t_ht['year'].astype(str)
            t_ht['created_at']= t_ht['created_at'].apply(lambda x : datetime.strptime(x, '%m-%Y'))

            fig = px.scatter(
                data_frame = t_ht,
                x = t_ht.set_index('created_at').index,
                y = 'hashtag',
                size = 'count',
                color = 'hashtag',
                opacity = 1,
                title = 'Top ten hashtags',
                labels = {'x':'date'},

            )
            fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False)
            fig.update_layout(plot_bgcolor='#060606', paper_bgcolor="#060606")
            fig.update_layout(
                                font_color="white",
                                title_font_color="white",
                                legend_title_font_color="white"
                            )
            fig.update_xaxes(tickangle=45, color='white', title = 'Date')
            return fig

@app.callback(
    dash.dependencies.Output('hashtag_choice', 'options'),
    [dash.dependencies.Input('state_actor', 'value'),
    dash.dependencies.Input('emb_country', 'value')]
)
def update_hash_dropdown(state_actor, emb_country):
    user_id = embassy_accounts[(embassy_accounts.state_actor == state_actor)
                               & (embassy_accounts.country == emb_country)].iloc[0][3]
    # query all hashtags pertaining to a user
    top_10, _ = get_top_ten(user_id)

    return [{'label': i, 'value': i} for i in top_10]

@app.callback(
    [Output('ner_network', 'elements'),
    Output('nb_hash', 'figure'),
    Output('hashtag-header','children')],
    [Input('state_actor', 'value'),
     Input('emb_country', 'value'),
     Input('hashtag_choice', 'value')
]
)

def update_network(state_actor, emb_country, hashtag_choice):
    if hashtag_choice is None:
        raise dash.exceptions.PreventUpdate()
    else:
        network_elements = get_network(state_actor, emb_country, hashtag_choice)
        hashtag_id = engine.execute(f"SELECT hashtag_id \
                    FROM hashtags \
                    WHERE hashtag = ('{hashtag_choice}')").fetchone()[0]
        hashtag_group =  pd.read_sql_query(f"SELECT * \
                                           FROM top_hash_bay \
                                        WHERE (hashtag_id = ('{hashtag_id}')) " , con=engine)
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Bar(name='hashtag', x=hashtag_group.date, y = hashtag_group['hashtag']),
            go.Bar(name='no hashtag', x=hashtag_group.date, y = hashtag_group['no_hashtag'])
        ])
        # Change the bar mode
        fig.update_layout(barmode='stack')
        fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False)
        fig.update_layout(plot_bgcolor='#060606', paper_bgcolor="#060606")
        fig.update_layout(
                            font_color="white",
                            title_font_color="white",
                            legend_title_font_color="white"
                        )
        fig.update_xaxes(tickangle=45, color='white', title = 'Date')
        title = f'Hashtag Co-occurrence - Embassy of {state_actor} in {emb_country}'
        return network_elements, fig, title
