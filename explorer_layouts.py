
"""
Created - July 2021
Author - caffsean

This file is the LAYOUT for the 'Explorer' component of the Embassy Twitter Dashboard
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
pd.options.display.max_columns = None

# Token for mapbox
token = 'pk.eyJ1IjoiY2FmZnNlYW4iLCJhIjoiY2twa2Z6NHpmMGZobDJwbmc4ZGN1dW9uaCJ9.3ILZmNsg-1PZhNSb9hKgrw'


external_stylesheets = [dbc.themes.JOURNAL]
### Hardcoded Dropdown Variables
indicators = ['Emojis','User Mentions','Hashtags','Language']
networks = ['Russia','United States of America']
user_attributes = {'statuses_count':'Statuses Count','followers_count':'Followers Count'}
timeseries = ['Tweet Frequency','Likes','Retweets']
timeseries_embassy = ['count','favorite_count','retweet_count','retweet_bool']
token_type = ['emojis','hashtags','retweeted_users','user_mentions','language','frequency_data']
barchart_options = ['top_emojis', 'top_hashtags', 'top_user_mentions', 'top_language_use', 'spacy_pos_en_ADJ', 'spacy_pos_en_VERB', 'spacy_pos_en_NOUN', 'spacy_pos_en_PROPN', 'spacy_pos_en_ADV', 'spacy_pos_ru_ADJ', 'spacy_pos_ru_VERB', 'spacy_pos_ru_NOUN', 'spacy_pos_ru_PROPN', 'spacy_pos_ru_ADV', 'spacy_pos_fr_ADJ', 'spacy_pos_fr_VERB', 'spacy_pos_fr_NOUN', 'spacy_pos_fr_PROPN', 'spacy_pos_fr_ADV', 'spacy_pos_es_ADJ', 'spacy_pos_es_VERB', 'spacy_pos_es_NOUN', 'spacy_pos_es_PROPN', 'spacy_pos_es_ADV', 'spacy_ent_en_PERSON', 'spacy_ent_en_NORP', 'spacy_ent_en_FAC', 'spacy_ent_en_ORG', 'spacy_ent_en_GPE', 'spacy_ent_en_LOC', 'spacy_ent_en_PRODUCT', 'spacy_ent_en_EVENT', 'spacy_ent_en_LAW', 'spacy_ent_en_LANGUAGE', 'spacy_ent_en_DATE', 'spacy_ent_ru_PERS', 'spacy_ent_ru_NORP', 'spacy_ent_ru_FAC', 'spacy_ent_ru_ORG', 'spacy_ent_ru_GPE', 'spacy_ent_ru_LOC', 'spacy_ent_ru_PRODUCT', 'spacy_ent_ru_EVENT', 'spacy_ent_ru_LAW', 'spacy_ent_ru_LANGUAGE', 'spacy_ent_ru_DATE', 'spacy_ent_fr_PERSON', 'spacy_ent_fr_NORP', 'spacy_ent_fr_FAC', 'spacy_ent_fr_ORG', 'spacy_ent_fr_GPE', 'spacy_ent_fr_LOC', 'spacy_ent_fr_PRODUCT', 'spacy_ent_fr_EVENT', 'spacy_ent_fr_LAW', 'spacy_ent_fr_LANGUAGE', 'spacy_ent_fr_DATE', 'spacy_ent_es_PERSON', 'spacy_ent_es_NORP', 'spacy_ent_es_FAC', 'spacy_ent_es_ORG', 'spacy_ent_es_GPE', 'spacy_ent_es_LOC', 'spacy_ent_es_PRODUCT', 'spacy_ent_es_EVENT', 'spacy_ent_es_LAW', 'spacy_ent_es_LANGUAGE', 'spacy_ent_es_DATE']
extended_barchart_options = barchart_options + ['frequency_data']
frequency_options = {'frequency_tweets':'Frequency of Tweets', 'frequency_retweets':'Frequency of Retweets', 'favorites_counts':'Favorite Counts', 'retweets_counts': 'Retweeted Counts','sentiment':'Vader Sentiment Analysis'}

events = pd.read_csv('assets/events.csv')
event_options = list(events['event'].values)

large_text_white = {'font-family': "Helvetica",'font-weight': '300','font-size': '300%','color': '#F2F2F2'}
large_text_black = {'font-family': "Helvetica",'font-weight': '200','font-size': '300%','color': '#111111'}
medium_text_white = {'font-family': "Helvetica",'font-weight': '110','font-size': '200%','color': '#F2F2F2'}
small_text_white = {'font-family': "Helvetica",'font-weight': '100','font-size': '100%','color': '#F2F2F2'}

style_div_black = {'backgroundColor': 'black','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}
style_div_steel = {'backgroundColor': 'steelblue','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}

dropdown_style = {'color': 'blue','background-color': '#bebebe'}
graph_background = {'backgroundColor': '#22303D','padding': '10px 10px 10px 10px','border':'4px solid', 'border-radius': 10}
graph_background_blue = {'backgroundColor': '#2D455C','padding': '10px 10px 10px 10px','border':'4px solid','border-color':'black','border-radius': 10}
card_style = {'border-radius':10}

## Layout of the 'Cards' that display text info

card_content_0 = [
    dbc.CardBody(
        [
            html.H1(id='card-0-number'),
            html.H4('Total Embassies'),
        ]
    ),
]

card_content_1 = [
    dbc.CardBody(
        [
            html.H1(id='card-1-number', className="card-title"),
            html.H4('Embassy Twitter Accounts', className="card-title"),
        ]
    ),
]

card_content_2 = [
    dbc.CardBody(
        [
            html.H1(id='card-2-number', className="card-title"),
            html.H4('Total Followers', className="card-title"),
        ]
    ),
]

card_content_3 = [
    dbc.CardBody(
        [
            html.H1(id='card-3-number', className="card-title"),
            html.H4('Total Statuses', className="card-title"),
        ]
    ),
]

card_content_network = [
    dbc.CardBody(
        [
            html.H1(id='main_card', className="card-title"),
            html.H4('info', className="card-title"),
            html.H4('info', className="card-title"),
            html.H4('info', className="card-title"),
            html.H4('info', className="card-title"),
            html.H4('info', className="card-title"),
            html.H4('info', className="card-title"),
            html.H4('info', className="card-title"),
        ]
    ),
]


card_content_embassy = [
    dbc.CardBody(
        [
            html.H1(id='embassy-name-card', className="card-text"),
            html.H2(id='embassy-screen_name', className="card-text",style={'color':'blue'}),
            html.Br(),
            html.H4(id='embassy-description', className="card-text"),
            html.H4(id='embassy-created_at', className="card-text"),
            html.H4(id='embassy-statuses_count', className="card-text"),
            html.H4(id='embassy-samples', className="card-text"),
            html.H4(id='embassy-ratio-retweets', className="card-text"),
            html.H4(id='embassy-ratio-original', className="card-text"),
        ]
    ),
]



card_content_compare = [
    dbc.CardBody(
        [
            html.H3('Pearson correlation coefficient:', className="card-title",style={'color':'blue'}),
            html.H1(id='correlation', className="card-title"),
            html.H3(id='p-value', className="card-title"),
            html.H4('* score of 1 is "perfect" correlatiom', className="card-title"),
            html.Br(),
            html.H3('Dynamic Time Warping Score:', className="card-title",style={'color':'blue'}),
            html.H1(id='dtw-score', className="card-title"),
            html.H4('* score of 0 is "perfect" similarity', className="card-title"),
        ]
    ),
]

## LAYOUT OF NETWORK LEVEL
layout = html.Div([

    ### HEADER
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H1('Twitter Diplomacy Dashboard: Network Level',style=large_text_white),
                        ],width=11),
                        dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dbc.Button("About", id="popover-network-button", color="info"),
                        ])],color='dark',style=card_style),
                        dbc.Popover(
                        [dbc.PopoverHeader("The Network-Level Dashboard:"),
                        dbc.PopoverBody("This component of the dashboard allows users to explore Twitter diplomacy at the network level. That is, users can examine the diplomatic footprint of a nation in aggregate. "),
                                ],
                                id="popover-network-info",
                                target="popover-network-button",  # needs to be the same as dbc.Button id
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
            dbc.Row([
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                    html.H3('Choose a diplomatic network to explore:',style=small_text_white),
                    dcc.Dropdown(id='dropdown-network',
                                style = dropdown_style,
                                value='Russia',
                                options=[{'label': emb, 'value': emb} for emb in networks]),
                                            ])],color='dark',style=card_style)
                        ],width=5)
                    ])
                ],style=graph_background_blue),
            ],style=style_div_black),

    html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H1(id='header-h1-network',style=large_text_white)
                        ])
                    ]),
            dbc.Row([
                    dbc.Col(dbc.Card(card_content_0, color="dark",inverse=True,style=card_style)),
                    dbc.Col(dbc.Card(card_content_1, color="dark",inverse=True,style=card_style)),
                    dbc.Col(dbc.Card(card_content_2, color="dark",inverse=True,style=card_style)),
                    dbc.Col(dbc.Card(card_content_3, color="dark",inverse=True,style=card_style)),
                    ]),
                ],style=graph_background),
        html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-bubble',
                                        value='statuses_count',
                                        style=dropdown_style,
                                        options=[{'label': v, 'value': k} for k,v in user_attributes.items()])
                                    ],width=8),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-network-timeseries',
                                        multi=True,
                                        style=dropdown_style,
                                        value=['frequency_tweets','frequency_retweets'],
                                        options=[{'label': v, 'value': k} for k,v in frequency_options.items()])
                                    ],width=4),
                                ]),
                            ]),
                        ]),
                    ],style=graph_background_blue),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H4(''),
                    dbc.Card([dbc.CardBody([
                    dcc.Graph(id='graph-big-map',clickData={'points': [{'customdata': 'CANADA'}]})
                    ])],color='dark',style=card_style)
                        ],width=8),
                dbc.Col([
                    html.Br(),
                    html.H4(''),
                    dbc.Card([dbc.CardBody([
                    dcc.Graph(id='graph-timeseries-network')
                    ])],color='dark',style=card_style)
                        ],width=4),
                     ]),
            ],style=graph_background),

    ### Row 6
    ### Dropdowns for Network Level
        html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(id='dropdown-network-barchart',
                            style=dropdown_style,
                            value=barchart_options[0],
                            options=[])
                        ],width=4),
                dbc.Col([
                        ],width=3),
                    ]),
                ],style=graph_background_blue),
    ## Row 7
    ## Graphs for Network Level
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                    dcc.Graph(id='graph-barchart-network',
                            clickData={})
                    ])],color='dark',style=card_style)
                        ],width=3),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                    dcc.Graph(id='graph-heatmap-network')
                    ])],color='dark',style=card_style)
                        ],width=6),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                    dcc.Graph(id='graph-top-countries-network')
                    ])],color='dark',style=card_style)
                        ],width=3),
                    ]),
                ],style=graph_background),
    ],style=style_div_black),
])



### LAYOUT FOR EMBASSY LEVEL
layout_2 = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H1('Twitter Diplomacy Dashboard: Embassy Level',style=large_text_white),
                        ],width=11),
                        dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dbc.Button("About", id="popover-embassy-button", color="info"),
                        ])],color='dark',style=card_style),
                        dbc.Popover(
                        [dbc.PopoverHeader("The Embassy-Level Dashboard:"),
                        dbc.PopoverBody("This component of the dashboard allows users to explore particular embassies. Click on the heatmap to see the original tweets! "),
                                ],
                                id="popover-embassy-info",
                                target="popover-embassy-button",  # needs to be the same as dbc.Button id
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
        ## Row 8
        ## Dropdown for Embassy Level
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H1(id='header-h1-embassy',style=large_text_white)
                        ],width=7),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                    dcc.Dropdown(id='dropdown-network',
                                style=dropdown_style,
                                value='Russia',
                                options=[{'label': emb, 'value': emb} for emb in networks]),
                    dcc.Dropdown(id='dropdown-embassy',
                                style=dropdown_style,
                                value=None,
                                options=[])
                    ])],color='dark',style=card_style)
                        ],width=4),
                # dbc.Col([
                #     dbc.Button("About", id="popover-embassy-button-2", color="info"),
                #     dbc.Popover(
                #     [dbc.PopoverHeader("Info:"),
                #     dbc.PopoverBody("info"),
                #             ],
                #             id="popover-embassy-info-2",
                #             target="popover-embassy-button-2",  # needs to be the same as dbc.Button id
                #             placement="bottom",
                #             is_open=False,
                #         ),
                #         ],width=1)
                    ]),
                ],style=graph_background_blue),
        ## Row 9
        ## Graphs for Embassy Level
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([],width=3),
                    dbc.Col([
                            dcc.Dropdown(id='dropdown-timeseries-embassy',
                                    style=dropdown_style,
                                    multi=True,
                                    value=['frequency_tweets','frequency_retweets'],
                                    options=[{'label': v, 'value': k} for k,v in frequency_options.items()])
                                    ],width=6)
                        ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(card_content_embassy, color="dark",inverse=True,style=card_style),
                            ],width=3),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dcc.Graph(id='graph-timeseries-embassy')
                        ])],color='dark',style=card_style)
                            ],width=6),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dcc.Graph(id='graph-wordcloud-embassy', figure={}, config={'displayModeBar': False}),
                        ])],color='dark',style=card_style)
                            ],width=3),
                        ]),
                    ],style=graph_background),

            ## Row 10
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='dropdown-token-embassy',
                                    multi=False,
                                    style=dropdown_style,
                                    value='',
                                    options=[])
                                    ],width=3)
                        ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dcc.Graph(id='graph-barchart-embassy',
                                    clickData={'points': [{'customdata': []}]})
                        ])],color='dark',style=card_style)
                            ],width=3),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dcc.Graph(id='graph-heatmap-embassy',
                                    clickData={'points': [{'customdata': ['2020-11-30','🇷🇺']}]})
                        ])],color='dark',style=card_style)
                            ],width=6),
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                        dcc.Graph(id='graph-textmarkdown-embassy')
                        ])],color='dark',style=card_style)
                            ],width=3)
                        ]),
                    ],style=graph_background),
            ],style=graph_background_blue),
    ],style=style_div_black),
])


### LAYOUT FOR COMPARATIVE TIME SERIES
layout_3 = html.Div([

    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H1('Comparative Time Series',style=large_text_white),
                        ],width=11),
                        dbc.Col([
                            dbc.Button("About", id="popover-comparative-button", color="info"),
                            dbc.Popover(
                            [dbc.PopoverHeader("Comparative Time Series:"),
                            dbc.PopoverBody("The comparative time series dashboard allows users to compare the use of tokens over time or other frequency data across all embassies. "),
                                    ],
                                    id="popover-comparative-info",
                                    target="popover-comparative-button",  # needs to be the same as dbc.Button id
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
        ## Row 8
        ### All those dropdowns
        html.Div([
            # dbc.Row([
            #     dbc.Col([
            #         html.H1('Time Series Analysis',style=large_text_white)
            #             ],),
            #         ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                    ],width=2),
                            dbc.Col([
                                html.H4('Network',style=small_text_white)
                                    ],width=2),
                            dbc.Col([
                                html.H4('Embassy',style=small_text_white)
                                    ],width=2),
                            dbc.Col([
                                html.H4('Token Type',style=small_text_white)
                                    ],width=3),
                            dbc.Col([
                                html.H4('Token',style=small_text_white)
                                    ],width=2),
                                ]),
                        dbc.Row([
                            dbc.Col([
                                html.H4('Filter Embassy Data:',style=medium_text_white)
                                    ],width=2),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-compare-network-1',
                                            clearable=False,
                                            value='Russia',
                                            options=[{'label': emb, 'value': emb} for emb in networks])
                                    ],width=2),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-compare-embassy-1',
                                            clearable=False,
                                            value='',
                                            options=[])
                                    ],width=2),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-compare-indicator-1',
                                            multi=False,
                                            clearable=False,
                                            value='',
                                            options=[])

                                    ],width=3),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-compare-token-1',
                                            multi=False,
                                            clearable=False,
                                            value='',
                                            options=[])
                                    ],width=2),
                                ]),
                            ])],color='dark',style=card_style),

                    dbc.Card([dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H4('Filter Embassy Data:',style=medium_text_white)
                                    ],width=2),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-compare-network-2',
                                            clearable=False,
                                            value='United States of America',
                                            options=[{'label': emb, 'value': emb} for emb in networks])
                                    ],width=2),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-compare-embassy-2',
                                            value='',
                                            clearable=False,
                                            options=[])
                                    ],width=2),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-compare-indicator-2',
                                            multi=False,
                                            clearable=False,
                                            value='',
                                            options=[])

                                    ],width=3),
                            dbc.Col([
                                dcc.Dropdown(id='dropdown-compare-token-2',
                                            multi=False,
                                            clearable=False,
                                            value='',
                                            options=[])
                                    ],width=2),
                                ])
                        ])],color='dark',style=card_style)
                    ],width=12)
                ])
                ],style=graph_background_blue),
        ## Row 9
        ## Timeseries Graph
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        dcc.Graph(id='graph-timeseries-compare')
                            ])],color='dark',style=card_style),
                        ],width=9),
                dbc.Col([
                    dbc.Card(card_content_compare, color="dark", inverse=True, style=card_style),
                        ],width=3),
                    ]),
                ],style=graph_background),

        ## Row 10
        ## Historical Event Dropdown
        html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(id='historical-event-dropdown',
                                multi=False,
                                placeholder='Choose a historical event',
                                value="",
                                options=[{'label': ev, 'value': ev} for ev in event_options])
                        ],width=3)
                    ]),
                ],style=style_div_steel),
    ],style=style_div_black),
    html.Div([
    html.P(id='output') ## For clickback data
    ]),
    ]) ## Last Div
