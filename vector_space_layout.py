
"""
Created - July 2021
Author - caffsean
"""

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import pandas as pd
import dash_daq as daq
import dash_html_components as html
pd.options.display.max_columns = None
from my_dictionaries import language_dictionary,label_dictionary
import word_network_tools
from gensim.models import Word2Vec

us_model = Word2Vec.load("assets/language_models/us_en_w2v.model")
rus_model = Word2Vec.load("assets/language_models/rus_en_w2v.model")


large_text_white = {'font-family': "Helvetica",'font-weight': '300','font-size': '300%','color': '#F2F2F2'}
large_text_black = {'font-family': "Helvetica",'font-weight': '200','font-size': '300%','color': '#111111'}
medium_text_white = {'font-family': "Helvetica",'font-weight': '110','font-size': '200%','color': '#F2F2F2'}
small_text_white = {'font-family': "Helvetica",'font-weight': '100','font-size': '120%','color': '#F2F2F2'}

style_div_black = {'backgroundColor': 'black','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}
style_div_purple = {'backgroundColor': 'purple','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}
style_div_thistle = {'backgroundColor': 'thistle','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}
style_div_steel = {'backgroundColor': 'steelblue','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}
style_div_slate = {'backgroundColor': '#5D6D7E','padding': '20px 20px 20px 20px','border':'1px solid', 'border-radius': 10}
style_div_gray = {'backgroundColor': '#85929E','padding': '20px 20px 20px 20px','border':'1px solid', 'border-radius': 10}

dropdown_style = {'color': '#bebebe','background-color': '#bebebe'}
graph_background = {'backgroundColor': '#22303D','padding': '10px 10px 10px 10px','border':'4px solid', 'border-radius': 10}
graph_background_blue = {'backgroundColor': '#2D455C','padding': '10px 10px 10px 10px','border':'4px solid','border-color':'black','border-radius': 10}
card_style = {'border-radius':10}
external_stylesheets = [dbc.themes.JOURNAL]

vector_layout = html.Div([

## TOP
html.Div([
    html.Div([
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H1('Document Embeddings of Embassy Twitter Accounts',style=large_text_white),
                        html.H1('Understanding Messaging WITHIN Countries',style=medium_text_white)
                            ],width=11),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Button("About", id="popover-document-button", color="info"),
                                        ])
                                ],color='dark',style=card_style),
                        dbc.Popover(
                            [dbc.PopoverHeader("Document Similarity Dashboard:"),
                            dbc.PopoverBody("This component of the dashboard allows users to examine the similarities and differences of messaging by particular actors WITHIN particular countries.")],
                                id="popover-document-info",
                                target="popover-document-button",  # needs to be the same as dbc.Button id
                                placement="bottom",
                                is_open=False),
                            ],width=1)
                        ])
                    ])
                ])
            ],style=graph_background)
        ],style=graph_background_blue),

## Document Similarity Vector Graph
html.Div([
    html.Div([
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H2(id='pca-header',style=large_text_white),
                        html.Br(),
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H4('Slide to change number of clusters:',style=small_text_white),
                                            ]),
                                        ]),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Slider(
                                            id='cluster-slider',
                                            min=0,
                                            max=10,
                                            value=2,
                                            marks={str(num): str(num) for num in list(range(0,10))},
                                            step=None
                                            )
                                            ])
                                        ])
                                ])],color='dark',style=card_style)
                            ],width=8),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H3('Customize Document Scatterplot',style=medium_text_white)
                                        ])
                                    ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.H4('Show BERT PCA:',style=small_text_white),
                                        html.H4('Color Continents:',style=small_text_white),
                                        html.H4('Show Text:',style=small_text_white),
                                        html.H4('Show 3-Dimensional:',style=small_text_white),
                                        html.H4('Show Clusters:',style=small_text_white),
                                    ]),
                                    dbc.Col([
                                        daq.BooleanSwitch(
                                          on=False,
                                          #label=dict(label='Show BERT PCA:',style=small_text_white),
                                          labelPosition="right",
                                          color='green',
                                          id='bert-switch'
                                        ),
                                        daq.BooleanSwitch(
                                          on=False,
                                          #label=dict(label='Color Continents:',style=small_text_white),
                                          labelPosition="left",
                                          color='green',
                                          id='color-switch'
                                        ),
                                        daq.BooleanSwitch(
                                          on=False,
                                          #label=dict(label='Show Text:',style=small_text_white),
                                          labelPosition="left",
                                          color='green',
                                          id='text-switch'
                                        ),
                                        daq.BooleanSwitch(
                                          on=False,
                                          #label=dict(label='Show 3-Dimensional:',style=small_text_white),
                                          labelPosition="left",
                                          color='green',
                                          id='dim-switch'
                                        ),
                                        daq.BooleanSwitch(
                                          on=False,
                                          #label=dict(label='Show Clusters:',style=small_text_white),
                                          labelPosition="left",
                                          color='green',
                                          id='k-switch')
                                            ]),
                                        ]),
                                    ])],color='dark',style=card_style),
                            ],width=4),
                        ])
                    ])
                ])
            ],style=graph_background)
        ],style=graph_background_blue),

## Graph

html.Div([
    html.Div([
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([dbc.CardBody(
                        dcc.Graph(id='graph-vector-space',),#style={'width': '100vh', 'height': '80vh'}),
                        )],color="dark",style=card_style)
                    ],width=12),

                    ])
                ])
            ])
        ],style=graph_background)
    ],style=graph_background_blue),


### Insights from BERT
### Embedding Vector Space
html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H2('Document Similarity with BERT',style=large_text_white)
                    ],width=8),
            dbc.Col([
                dbc.Card([dbc.CardBody([
                        html.H2(id='header_embassy_vector_space',style=large_text_white),
                        html.H3('Choose a diplomatic network to explore:',style=small_text_white),
                        dcc.Dropdown(id='dropdown-network-BERT',
                                    value='United States of America',
                                    options=[{'label': net, 'value': net} for net in ['United States of America','Russia']]),
                        dcc.Dropdown(id='dropdown-embassy-BERT',
                                    value='',
                                    options=[]),
                        ])],color='dark',style=card_style)
                    ],width=3),
            dbc.Col([
                dbc.Card([dbc.CardBody([
                dbc.Button("About", id="popover-document-button-2", color="info")])],color='dark',style=card_style),
                dbc.Popover(
                [dbc.PopoverHeader("BERT:"),
                dbc.PopoverBody("BERT is a language model developed by Google. Transforming Twitter accounts into BERT vectors allows us to use mathematical tools to explore the semantic similarities and differences of entire embassy accounts."),
                        ],
                        id="popover-document-info-2",
                        target="popover-document-button-2",  # needs to be the same as dbc.Button id
                        placement="bottom",
                        is_open=False,
                    ),
                    ],width=1)
            ])
        ],style=graph_background_blue),

        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4(id='bert-similarity-header',style=medium_text_white),
                        html.Br(),
                        html.H4('Which embassy Twitter accounts are most similar to a given embassy account?', style=small_text_white)
                    ],width=4),
                    dbc.Col([
                        html.H4(id='bert-outliers-header',style=medium_text_white),
                        html.Br(),
                        html.Br(),
                        html.H4("Where is a given diplomatic network's messaging the most unusual?", style=small_text_white)
                    ],width=4),
                    dbc.Col([
                        html.H4('Most Divergent Accounts',style=medium_text_white),
                        html.Br(),
                        html.H4("In which countries do American and Russian embassy accounts diverge the most in terms of their messaging?", style=small_text_white)
                    ],width=4)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                            dcc.Graph(id='bert-similarity',style={})
                        ])],color="dark",style=card_style)
                    ],width=4),
                    dbc.Col([
                        dbc.Card([dbc.CardBody(
                        dcc.Graph(id='bert-outliers')
                        )],color="dark",style=card_style)
                    ],width=4),
                    dbc.Col([
                        dbc.Card([dbc.CardBody(
                        dcc.Graph(id='bert-divergence')
                        )],color="dark",style=card_style)
                    ],width=4)
                ])
            ],style=graph_background)
        ],style=graph_background_blue)
],style=graph_background),

])

### Embedding Vector Space
vector_layout_2 = html.Div([

html.Div([
    html.Div([
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H1('Word Embeddings of Embassy Twitter Corpora',style=large_text_white),
                        html.H1('Understanding Messaging ABOUT Countries',style=medium_text_white)
                    ],width=11),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dbc.Button("About", id="popover-word-button", color="info"),])],color='dark',style=card_style),
                        dbc.Popover(
                        [dbc.PopoverHeader("Word Embeddings Dashboard:"),
                        dbc.PopoverBody("This component of the dashboard allows users to examine the similarities and differences of messaging ABOUT particular countries."),
                                ],
                                id="popover-word-info",
                                target="popover-word-button",  # needs to be the same as dbc.Button id
                                placement="bottom",
                                is_open=False,
                            ),
                    ],width=1)
                ])
            ])
        ])
    ],style=graph_background)
],style=graph_background_blue),

html.Div([
    html.Div([
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H2(id='header_embassy_vector_space',style=large_text_white),
                        ],width=8),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        html.H3('Choose a diplomatic network to explore:',style=small_text_white),
                        dcc.Dropdown(id='dropdown-network',
                                    style=dropdown_style,
                                    value='United States of America',
                                    options=[{'label': net, 'value': net} for net in ['United States of America','Russia']]),
                        ])],color='dark',style=card_style)
                            ],width=4)
                    ]),
                ])
            ],style=graph_background)
        ],style=graph_background_blue),


        html.Div([
            html.Div([
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dcc.Graph(id='graph-country-vectors',
                        style={})
                        ])],color='dark',style=card_style)
                    ],width=8),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dcc.Graph(id='mahanalobis-table')
                        ])],color='dark',style=card_style)
                    ],width=4)
                ])
            ])
        ],style=graph_background),
    ],style=graph_background_blue),


html.Div([
    html.Div([
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H2(id='header_search_vector_space',style=large_text_white),
                            ],width=8),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        html.H3('Choose a diplomatic network to explore:',style=small_text_white),
                        dcc.Dropdown(id='dropdown-network-search',
                                    value='United States of America',
                                    style=dropdown_style,
                                    options=[{'label': net, 'value': net} for net in ['United States of America','Russia']]),
                        html.H4('Search:',style=small_text_white),
                        dcc.Input(id="search",
                                    debounce=True,
                                    value='China'),
                        ])],color='dark',style=card_style),
                            ],width=3),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dbc.Button("About", id="popover-word-button-2", color="info"),])],color='dark',style=card_style),
                        dbc.Popover(
                        [dbc.PopoverHeader("Info:"),
                        dbc.PopoverBody("info"),
                                ],
                                id="popover-word-info-2",
                                target="popover-word-button-2",  # needs to be the same as dbc.Button id
                                placement="bottom",
                                is_open=False,
                            ),
                            ],width=1),
                        ]),
                    ])
                ],style=graph_background)
            ],style=graph_background_blue),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        dcc.Graph(id='graph-explore-vectors',
                        style={})
                    ])],color='dark',style=card_style)
                ],width=12),
            ])
        ],style=graph_background)
    ],style=graph_background_blue),
],style=graph_background)

 ## final
