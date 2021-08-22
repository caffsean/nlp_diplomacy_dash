'''
Topic Modeling - Layout

'''

import pandas as pd
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.io as pio
pio.renderers.default = 'iframe'
from country2ids import rus_country2id,us_country2id,us_languages,rus_language


US_data_to_show = pd.read_csv('assets/USA_topic_log.csv')
us_ids = list(US_data_to_show.ids.unique())
us_years = list(US_data_to_show.years.unique())
us_years = [str.strip(x) for x in us_years]
us_years.sort()

RUS_data_to_show = pd.read_csv('assets/RUS_topic_log.csv')
rus_ids = list(RUS_data_to_show.ids.unique())
rus_years = list(RUS_data_to_show.years.unique())
rus_years = [str.strip(x) for x in rus_years]
rus_years.sort()

years = set(rus_years+us_years)
years = sorted(list(years))

style_div_black = {'backgroundColor': 'black','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}
style_div_purple = {'backgroundColor': 'purple','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}
style_div_thistle = {'backgroundColor': 'thistle','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}
style_div_steel = {'backgroundColor': 'steelblue','padding': '10px 10px 10px 10px','border':'1px solid', 'border-radius': 10}



large_text_white = {'font-family': "Helvetica",'font-weight': '300','font-size': '300%','color': '#F2F2F2'}
large_text_black = {'font-family': "Helvetica",'font-weight': '200','font-size': '300%','color': '#111111'}
medium_text_white = {'font-family': "Helvetica",'font-weight': '110','font-size': '200%','color': '#F2F2F2'}
small_text_white = {'font-family': "Helvetica",'font-weight': '100','font-size': '170%','color': '#F2F2F2'}

dropdown_style = {'color': 'blue','background-color': '#212121'}
graph_background = {'backgroundColor': '#22303D','padding': '10px 10px 10px 10px','border':'4px solid', 'border-radius': 10}
graph_background_blue = {'backgroundColor': '#2D455C','padding': '10px 10px 10px 10px','border':'4px solid','border-color':'black','border-radius': 10}
card_style = {'border-radius':10}


external_stylesheets = [dbc.themes.JOURNAL]



redo_topic_layout = html.Div([

### Header
html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H1('Exploring Embassy Twitter Streams Through Topic Modeling',style=large_text_white)
            ],width=11),
            dbc.Col([
            dbc.Card([dbc.CardBody([
            dbc.Button("About", id="popover-topic-button", color="info"),
            ])],color='dark',style=card_style),
            dbc.Popover(
            [dbc.PopoverHeader("Topic Modeling Dashboard:"),
            dbc.PopoverBody("This component of the dashboard allows users to observe the results of topic modeling techniques in order to determine which topics are prevalent in a country’s diplomatic network as well as individual embassy accounts."),
                    ],
                    id="popover-topic-info",
                    target="popover-topic-button",  # needs to be the same as dbc.Button id
                    placement="bottom",
                    is_open=False,
                ),
            ],width=1)
        ])
    ],style=graph_background)
],style=graph_background_blue),

html.Div([
    html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label('Select Number of Topics',style=medium_text_white),
                            html.Br(),
                            dcc.Slider(
                                id='opt1-numtopic',
                                min=2,max=9,step=1,value=9,
                                marks={2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'})
                    ])],color='dark',style=card_style),
                    html.Br(),
                    html.H3(id='topic-title',style=large_text_white)
                    ],width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H3('Search Topic Models',style=medium_text_white),
                                    html.Br(),
                                    ])
                                ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H4('Select Network:',style=small_text_white),
                                    html.H4('Select Country:',style=small_text_white),
                                    html.H4('Select Language:',style=small_text_white),
                                    html.H4('Select Year:',style=small_text_white),
                                ]),
                                dbc.Col([
                                    dcc.Dropdown(id='network_dropdown',
                                                options=[{'label':k,'value':k} for k in ['Russia', 'United States']],
                                                value='United States'),
                                    dcc.Dropdown(id='opt2-embID',
                                                 value='ALL'),
                                    dcc.Dropdown(id='opt3-lang',
                                                options= [{'label':k, 'value':v} for k,v in us_languages.items()],
                                                value='en'),
                                    dcc.Dropdown(
                                                id='opt4-year',
                                                options = [{'label':i,'value':i} for i in years],
                                                value='2021'),
                                        ]),
                                    ]),
                                ])],color='dark',style=card_style),
                        ],width=4)
                    ])
                ],style=graph_background),
        html.Br(),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([dbc.CardBody([
                                dcc.Graph(id='us_topic')
                            ])],color='dark',style=card_style)
                        ])
                    ])
        ],style=graph_background_blue)
    ],style=graph_background),
],style=graph_background_blue),


]) ## Last Bracket




topic_layout = html.Div([
html.Div([
    html.Div([
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H1('Exploring embassy twitter topics throughout time and space',style={}),
                    ],width=12)
                ])
            ])
        ])
    ],style=style_div_thistle)
],style=style_div_black),

# need a slider
html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Br(),
                html.H3('Select number of topics to explore'),
                html.Br(),
                # html.Br(),
                    dcc.Slider(
                        id='opt1-numtopic',
                        min=2,max=6,step=1,value=6,
                        marks={2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'})
            ])
        ])
    ])
]),
# US fig
html.Div([
    html.Div([
        html.H4('Exploring US topics...'),
        html.Div([
            html.Div([html.Label('Select all or a specific embassy:'),
                html.Div([
                    dcc.Dropdown(
                        id='opt2-embID',
                        options=[{'label':k,'value':v} for k,v in us_country2id.items()],
                        value='all'
                    ),
                    html.Div([html.Label('Select all or specific language:'),

                        html.Div([
                            dcc.Dropdown(
                                        id='opt3-lang',
                                        options= [{'label':k, 'value':v} for k,v in us_languages.items()],
                                        value='en'
                                    ),
                        ])
                    ])
                ])
            ])
        ])
    ])
]),
html.Div([
    html.Div([html.Label('Select all or a particular year:'),
        html.Div([
            dcc.Dropdown(
                id='opt4-year',
                options = [{'label':i,'value':i} for i in us_years],
                value='2021'
            )
        ])
    ])
]),

html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id='us_topic')
        ])
    ])
]),
html.Br(),

html.Div([
    html.Div([html.H4('Exploring Russian embassy topics...'),
        html.Div([
            html.Label('Select all or a specific embassy:'),
            dcc.Dropdown(id='opt2-embID2',
                        options=[{'label':k,'value':v} for k,v in rus_country2id.items()],
                        value='all')
        ])
    ])
]),

html.Div([
    html.Div([
        html.Div([
            html.Label('Select all or a specific language:'),
            dcc.Dropdown(
                id='opt3-lang2',
                options=[{'label':k,'value':v} for k,v in rus_language.items()],
                value='en'
            )
        ])
    ])
]),
html.Div([
    html.Div([
        html.Div([
            html.Label('Select all or a specific year:'),
            dcc.Dropdown(
                id='opt4-year2',
                options=[{'label':i,'value':i} for i in rus_years],
                value='2021'
            )
        ])
    ])
]),
html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id='rus_topic')
        ])
    ])
])


                      ])
