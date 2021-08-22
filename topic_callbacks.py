'''
Topic Modeling - Callbacks

'''
import numpy as np
from dash.dependencies import Output, Input,State
import plotly.figure_factory as ff
import pickle as pkl
import plotly.io as pio
pio.renderers.default = 'iframe'
from country2ids import rus_country2id,us_country2id,us_languages,rus_language
from app import app

template = 'plotly_dark'

@app.callback(Output('opt2-embID','options'),
                Input('network_dropdown','value'))
# def update_emb_ids(network):
#     if network=='Russia':
#         return [{'label':k,'value':v} for k,v in rus_country2id.items()]
#     else:
#         return [{'label':k,'value':v} for k,v in us_country2id.items()]
def update_emb_ids(network):
    if network=='Russia':
        return [{'label':k,'value':k} for k in rus_country2id.keys()]
    else:
        return [{'label':k,'value':k} for k in us_country2id.keys()]



@app.callback([Output('us_topic','figure'),
                Output('topic-title','children')],
              [Input('opt1-numtopic','value'),
               Input('opt2-embID','value'),
               Input('opt3-lang','value'),
               Input('opt4-year','value'),
               Input('network_dropdown','value')
              ])
def update_US_graph(numtopic,embID,lang,year,network):
    if network == 'Russia':
        net = 'RUS'
        #main_options= [{'label':k,'value':v} for k,v in rus_country2id.items()]
        country_id = rus_country2id[embID]
        
        
    elif network == 'United States':
        net = 'USA'
        #main_options = [{'label':k,'value':v} for k,v in us_country2id.items()]
        country_id = us_country2id[embID]

    filetoload = f"assets/topic_data/{net}_{country_id}_{numtopic}_topics_{lang}_lang_{year}_year.pkl"
    with open(filetoload, "rb") as input_file:
        heatmap_data = pkl.load(input_file)
    labels = heatmap_data['label']
    df = heatmap_data['df']
    fig = ff.create_annotated_heatmap(np.array(df),y = list(df.index),annotation_text=np.array(labels), colorscale='Viridis')
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout(template=template)
    fig.update_layout(margin=dict(l=30, r=30, t=40, b=30))
    fig.update_layout(height=800)
    fig['layout']['yaxis']['autorange'] = "reversed"
    title = f'Topics of {network} in {embID}'
    return fig, title


@app.callback(Output('rus_topic','figure'),
              [Input('opt1-numtopic','value'),
               Input('opt2-embID2','value'),
               Input('opt3-lang2','value'),
               Input('opt4-year2','value')
              ])
def update_RUS_graph(numtopic,embID,lang,year):
    filetoload = f"assets/topic_data/RUS_{embID}_{numtopic}_topics_{lang}_lang_{year}_year.pkl"
    with open(filetoload, "rb") as input_file:
        heatmap_data = pkl.load(input_file)
    labels = heatmap_data['label']
    df = heatmap_data['df']
    fig = ff.create_annotated_heatmap(np.array(df),y = list(df.index),annotation_text=np.array(labels), colorscale='Viridis')
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout(template=template)
    return fig


@app.callback(
    Output("popover-topic-info", "is_open"),
    [Input("popover-topic-button", "n_clicks")],
    [State("popover-topic-info", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open
