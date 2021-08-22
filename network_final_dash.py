'''
Network Analysis - Named Entities - Callbacks and Layouts
'''


import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Output, Input
import pickle as pkl
import plotly.io as pio
pio.renderers.default = 'iframe'
from country2ids import rus_country2id,us_country2id,us_languages,rus_language
from app import app

# del rus_country2id['ALL']
# del us_country2id['ALL']

large_text_white = {'font-family': "Helvetica",'font-weight': '300','font-size': '300%','color': '#F2F2F2'}
large_text_black = {'font-family': "Helvetica",'font-weight': '200','font-size': '300%','color': '#111111'}
medium_text_white = {'font-family': "Helvetica",'font-weight': '110','font-size': '200%','color': '#F2F2F2'}
small_text_white = {'font-family': "Helvetica",'font-weight': '100','font-size': '170%','color': '#F2F2F2'}

dropdown_style = {'color': 'blue','background-color': '#212121'}
graph_background = {'backgroundColor': '#22303D','padding': '10px 10px 10px 10px','border':'4px solid', 'border-radius': 10}
graph_background_blue = {'backgroundColor': '#2D455C','padding': '10px 10px 10px 10px','border':'4px solid','border-color':'black','border-radius': 10}
graph_background_black = {'backgroundColor': 'black','padding': '20px 20px 20px 20px','border':'4px solid', 'border-radius': 10}
card_style = {'border-radius':10}

#_________________________Layout Objects_________________________#

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

us_network_measure_dropdown = dcc.Dropdown(id = 'centrality',
                                        options=[{'label':'DEGREE','value':'degree'},
                                                 {'label':'BETWEENNESS','value':'betweenness'},
                                                 {'label':'CLOSENESS','value':'closeness'}],
                                        value = 'degree',
                                        placeholder = "Select a centrality measure...")
rus_network_measure_dropdown = dcc.Dropdown(id = 'centrality2',
                                        options=[{'label':'DEGREE','value':'degree'},
                                                 {'label':'BETWEENNESS','value':'betweenness'},
                                                 {'label':'CLOSENESS','value':'closeness'}],
                                        value = 'degree',
                                        placeholder = "Select a centrality measure...")

card_content_us = [dbc.CardBody(
            [
                html.H4(id = 'us-trans', className="card-title"),
                html.H4(id='us-avg-clust', className="card-title"),
                html.H4(id='us-top-hubs', className="card-title"),
                html.H4(id='us-top-auth',className="card-title"),

            ]
        ),
    ]

card_content_rus = [dbc.CardBody(
            [
                html.H4(id = 'rus-trans', className="card-title"),
                html.H4(id='rus-avg-clust', className="card-title"),
                html.H4(id='rus-top-hubs', className="card-title"),
                html.H4(id='rus-top-auth',className="card-title"),

            ]
        ),
    ]

us_labels_dropdown = dcc.Dropdown(id='select-us',
                                  value = 'AFGHANISTAN')

us_emb_network = cyto.Cytoscape(id="us-network",
                                autoungrabify = False,
                                minZoom = 0.5,
                                maxZoom = 3,
                                layout={'name':'preset'},
                                style  = {'width': '100%', 'height':'1200px'},
                                stylesheet=default_stylesheet
                                )
rus_emb_network = cyto.Cytoscape(id="rus-network",
                                autoungrabify = False,
                                minZoom = 0.5,
                                maxZoom = 3,
                                layout={'name':'preset'},
                                style  = {'width': '100%', 'height':'1200px'},
                                stylesheet=default_stylesheet
                                )


rus_labels_dropdown = dcc.Dropdown(id='select-rus',
                                    options=[{'label':k,'value':v} for k,v in rus_country2id.items()],
                                    value = list(rus_country2id.values())[0])



#_________________________Layout definition_________________________#

network_layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H3('Network Analysis Dashboard',style=large_text_white),
                            html.Br(),
                                ],width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H3('Search entity networks',style=medium_text_white),
                                            html.Br(),
                                            ])
                                        ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.H4('Select Network:',style=small_text_white),
                                            html.H4('Select Country:',style=small_text_white),
                                            html.H4('Select Measure:',style=small_text_white),
                                        ]),
                                        dbc.Col([
                                            dcc.Dropdown(id='network-dropdown',
                                                        options=[{'label':k,'value':k} for k in ['Russia', 'United States']],
                                                        value='United States'),
                                            us_labels_dropdown,
                                            us_network_measure_dropdown,
                                                ]),
                                            ]),
                                        ])],color='dark',style=card_style),
                                ],width=4)
                            ]),
                        ],style=graph_background_blue),
                html.Br(),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H2(id='network-analysis-header',style=large_text_white)
                        ])
                    ])
                ],style=graph_background_blue),
                html.Br(),
                html.Div([
                    dbc.Row([dbc.Col([us_emb_network],width=9),
                             dbc.Col([dbc.Card(card_content_us, color="info", outline=True)],width=3)])
                    ],style=graph_background_black),




                ])
            ],style=graph_background_blue)
        ])
    ]),

#_________________________app callbacks_________________________#

@app.callback(Output('select-us','options'),
                Input('network-dropdown','value'))

def update_emb_ids(network):
    if network=='Russia':
        return [{'label':k,'value':k} for k in rus_country2id.keys()]
    else:
        return [{'label':k,'value':k} for k in us_country2id.keys()]



@app.callback([Output('us-network','elements'),
               Output('us-trans','children'),
              Output('us-avg-clust','children'),
              Output('us-top-hubs','children'),
              Output('us-top-auth','children'),
              Output('network-analysis-header','children')],
              [Input('select-us','value'),
               Input('centrality','value'),
               Input('network-dropdown','value')])

def update_us_graph(ids,measure,network):
    '''
    select by entity first and load necessary pickle files
    id - of the Embassy
    measure - Centrality measure
    network - Entity

    Output: graph_data - network elements for cytoscape
            trans - Transitivity score
            clcoef - Average Cluster Coefficient
            hubs - Five nodes with most nodes point to
            auths - Top five authority who points to other nodes
    '''
    if network == 'Russia':
        net = 'RUS'
        country_id = rus_country2id[ids]
    elif network == 'United States':
        net = 'USA'
        country_id = us_country2id[ids]

    filename = f'assets/network_data/{measure}/{net}_{country_id}_network_{measure}_new.pkl'
    graph_data = pkl.load(open(filename, "rb"))

    measure_data = pkl.load(open(f"assets/network_data/measures/{net}_{country_id}_network_{measure}.pkl",'rb'))

    trans = f"Transitivity score: {measure_data['transitivity']:.4f}"
    clcoef = f"Average cluster coeff: {measure_data['avg_clustering_coefficient']:.4f}"
    hubs = f"Top 5 hub nodes: {', '.join(measure_data['hub5'])}"
    auths = f"Top 5 authority nodes: {', '.join(measure_data['auth5'])}"
    title = f'Named Entity Co-occurrence: Embassy of {network} in {ids}'
    return graph_data,trans,clcoef,hubs,auths,title

## Update russian network - this is not necessary anymore

@app.callback([Output('rus-network','elements'),
               Output('rus-trans','children'),
              Output('rus-avg-clust','children'),
              Output('rus-top-hubs','children'),
              Output('rus-top-auth','children')],
              [Input('select-rus','value'),
               Input('centrality2','value')])

def update_us_graph(ids,measure):
    filename = f'assets/network_data/{measure}/RUS_{ids}_network_{measure}_new.pkl'
    graph_data = pkl.load(open(filename, "rb"))

    measure_data = pkl.load(open(f"assets/network_data/measures/RUS_{ids}_network_{measure}.pkl",'rb'))

    trans = f"Transitivity score: {measure_data['transitivity']:.4f}"
    clcoef = f"Average cluster coeff: {measure_data['avg_clustering_coefficient']:.4f}"
    hubs = f"Top 5 hub nodes: {','.join(measure_data['hub5'])}"
    auths = f"Top 5 authority nodes: {','.join(measure_data['auth5'])}"

    return graph_data,trans,clcoef,hubs,auths
