
"""
Created - July 2021
Author - caffsean

These are the callbacks for the 'Explorer' dashboard.
"""
from urllib.parse import parse_qs, unquote
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import dash
from scipy import stats
from plotly.subplots import make_subplots
from dash import callback_context
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State, ALL, ALLSMALLER, MATCH
from dash.exceptions import PreventUpdate
import ast
import re
from dash_table import DataTable
import pandas as pd
import numpy as np
import networkx as nx
import textwrap
from tslearn.metrics import dtw, dtw_path
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud


pd.options.display.max_columns = None
# for p in [plotly, dash, jd, dcc, html, dbc, pd, np]:
#     print(f'{p.__name__:-<30}v{p.__version__}')
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud

## Import app file to make the app run
from app import app
## Custom Dictionaries for Making Dropdowns
from my_dictionaries import language_dictionary,label_dictionary

# Token for mapbox
token = 'pk.eyJ1IjoiY2FmZnNlYW4iLCJhIjoiY2twa2Z6NHpmMGZobDJwbmc4ZGN1dW9uaCJ9.3ILZmNsg-1PZhNSb9hKgrw'

## DATA - Network/Embassy
#embassy_df = pd.read_csv('final_embassy_df.csv') ### Embassy Level Full
embassy_df = pd.read_csv('assets/embassy_db_SAMPLE.csv') ### Embassy Level SAMPLE
network_df = pd.read_csv('assets/edited_network_df.csv') ### Network Level
network_df['Base'] = ['United States of America', 'Russia']

events_df = pd.read_csv('assets/events.csv')
rus_top_tokens = pd.read_csv('assets/rus_top_tokens.csv').replace("BOLIVIA, PLURINATIONAL STATE OF", "BOLIVIA").replace("VENEZUELA, BOLIVARIAN REPUBLIC OF", "VENEZUELA").replace("UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND", "UNITED KINGDOM").set_index('Unnamed: 0')
us_top_tokens = pd.read_csv('assets/us_top_tokens.csv').replace("BOLIVIA, PLURINATIONAL STATE OF", "BOLIVIA").replace("VENEZUELA, BOLIVARIAN REPUBLIC OF", "VENEZUELA").replace("UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND", "UNITED KINGDOM").set_index('Unnamed: 0')

external_stylesheets = [dbc.themes.JOURNAL]
frequency_options = {'frequency_tweets':'Frequency of Tweets', 'frequency_retweets':'Frequency of Retweets', 'favorites_counts':'Favorite Counts', 'retweets_counts': 'Retweeted Counts','sentiment':'Vader Sentiment Analysis'}
barchart_options = ['top_emojis', 'top_hashtags', 'top_user_mentions', 'top_language_use', 'spacy_pos_en_ADJ', 'spacy_pos_en_VERB', 'spacy_pos_en_NOUN', 'spacy_pos_en_PROPN', 'spacy_pos_en_ADV', 'spacy_pos_ru_ADJ', 'spacy_pos_ru_VERB', 'spacy_pos_ru_NOUN', 'spacy_pos_ru_PROPN', 'spacy_pos_ru_ADV', 'spacy_pos_fr_ADJ', 'spacy_pos_fr_VERB', 'spacy_pos_fr_NOUN', 'spacy_pos_fr_PROPN', 'spacy_pos_fr_ADV', 'spacy_pos_es_ADJ', 'spacy_pos_es_VERB', 'spacy_pos_es_NOUN', 'spacy_pos_es_PROPN', 'spacy_pos_es_ADV', 'spacy_ent_en_PERSON', 'spacy_ent_en_NORP', 'spacy_ent_en_FAC', 'spacy_ent_en_ORG', 'spacy_ent_en_GPE', 'spacy_ent_en_LOC', 'spacy_ent_en_PRODUCT', 'spacy_ent_en_EVENT', 'spacy_ent_en_LAW', 'spacy_ent_en_LANGUAGE', 'spacy_ent_en_DATE', 'spacy_ent_ru_PERS', 'spacy_ent_ru_NORP', 'spacy_ent_ru_FAC', 'spacy_ent_ru_ORG', 'spacy_ent_ru_GPE', 'spacy_ent_ru_LOC', 'spacy_ent_ru_PRODUCT', 'spacy_ent_ru_EVENT', 'spacy_ent_ru_LAW', 'spacy_ent_ru_LANGUAGE', 'spacy_ent_ru_DATE', 'spacy_ent_fr_PERSON', 'spacy_ent_fr_NORP', 'spacy_ent_fr_FAC', 'spacy_ent_fr_ORG', 'spacy_ent_fr_GPE', 'spacy_ent_fr_LOC', 'spacy_ent_fr_PRODUCT', 'spacy_ent_fr_EVENT', 'spacy_ent_fr_LAW', 'spacy_ent_fr_LANGUAGE', 'spacy_ent_fr_DATE', 'spacy_ent_es_PERSON', 'spacy_ent_es_NORP', 'spacy_ent_es_FAC', 'spacy_ent_es_ORG', 'spacy_ent_es_GPE', 'spacy_ent_es_LOC', 'spacy_ent_es_PRODUCT', 'spacy_ent_es_EVENT', 'spacy_ent_es_LAW', 'spacy_ent_es_LANGUAGE', 'spacy_ent_es_DATE']
template = 'plotly_dark'


@app.callback(
            Output('card-0-number','children'),
            Output('card-1-number','children'),
            Output('card-2-number','children'),
            Output('card-3-number','children'),
            Output('header-h1-network','children'),
            [Input('dropdown-network','value')])
def update_cards(network):
    dff = embassy_df[embassy_df['Base']==network]
    facts = ('TRUE','Suspended')
    embassied = dff[dff['Embassy'].isin(facts)]
    value0 = len(embassied)
    value1 = len(embassied[embassied['HANDLE'] != '0'])
    value2 = dff['followers_count'].sum()
    value3 = dff['statuses_count'].sum()#np.round(dff['mean_tweets_per_week_mean'].mean(),2)
    big_title = "Diplomatic Twitter Network of {}".format(network)
    return value0, value1, value2, value3, big_title

## Update Embassy Choice Dropdown - (Prevents Countries that don't have data from appearing in dropdown)
@app.callback(
            Output('dropdown-embassy','options'),
            Output('dropdown-embassy','value'),
            [Input('dropdown-network','value')])
def update_country_dropdowns(network):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['HANDLE']!='0')]
    options_list = list(dff['Country'].values)
    options = [{'label': i, 'value': i} for i in options_list]
    value = options[0]['label']
    return options, value

### Update Network-Level Token Choice - (Prevents indicator tokens that don't have data from appearing in dropdown)
@app.callback(
            Output('dropdown-network-barchart','options'),
            [Input('dropdown-network','value')])
def update_network_token_dropdowns(network):
    dff2 = network_df[network_df['Base']==network]
    dff3 = pd.DataFrame(dff2.iloc[0])
    dff3 = dff3.reset_index()
    available_columns = dff3[dff3[1]!='{}'][dff3['index'].str.contains('timeseries')==False]['index'].values
    available_columns = [x for x in list(label_dictionary.keys()) if x in set(available_columns)]
    options = [{'label': label_dictionary[i],'value': i} for i in list(available_columns)]
    return options

### Update Embassy-Level Token Choice - (Prevents indicator tokens that don't have data from appearing in dropdown)
@app.callback([
                Output('dropdown-token-embassy','options'),
                Output('dropdown-token-embassy','value'),
                Input('dropdown-network','value'),
                Input('dropdown-embassy','value')])
def update_embassy_token_dropdowns(network,country):
    dff2 = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    dff3 = pd.DataFrame(dff2.iloc[0])
    dff3 = dff3.reset_index()
    columns = list(dff3.columns)
    available_columns = dff3[dff3[columns[1]]!='{}'][dff3['index'].str.contains('timeseries')==False]['index'].values
    available_columns = [x for x in list(label_dictionary.keys()) if x in set(available_columns)]
    options = [{'label': label_dictionary[i],'value': i} for i in list(available_columns)]
    value = options[0]['value']
    return options, value

## Barchart and Heatmap - Network Level - (Makes the bar chart and heatmap at the network level)
@app.callback(
            Output('graph-barchart-network','figure'),
            Output('graph-heatmap-network','figure'),
            Input('dropdown-network','value'),
            Input('dropdown-network-barchart','value')
            )
def network_barcharts(network,feature):
    dff = network_df[network_df['Base']==network]
    bar_data = dff[feature]
    heat_data = dff[feature+'_timeseries']
    dff1 = ast.literal_eval(bar_data.values[0])
    if feature == 'top_language_use':
        dff1 = {language_dictionary[k]:v for k,v in dff1.items()}
    x = list(dff1.keys())[:20]
    y = list(dff1.values())[:20]
    token_df = pd.DataFrame(x,columns=['key'])
    token_df['value'] = y
    token_df = token_df[::-1]
    fig1 = go.Figure([])
    barchart = go.Bar(y=token_df['key'],x=token_df['value'],orientation='h')
    fig1.add_trace(barchart)
    fig1.update_traces(customdata=token_df['key'])
    fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    fig1.update_layout(template=template)

    feature_name = label_dictionary[feature]
    title1 = f"{feature_name} of {network}'s Diplomatic Network"
    title2 = f"{feature_name} of {network} Over Time"

    dff2 = pd.DataFrame.from_dict(ast.literal_eval(heat_data.values[0])).reset_index()
    dff2 = dff2[list(dff2.columns)[:21]]
    list_of_columns = list(dff2.columns)[1:]
    list_of_columns = list_of_columns.reverse()
    heat_df = pd.melt(dff2, id_vars=['index'], value_vars=list_of_columns)
    heat_df = heat_df.replace(0, np.nan)
    if feature == 'top_language_use':
        heat_df['variable'] = [language_dictionary[k] for k in heat_df['variable']]
    heatmap = go.Heatmap(
        z=heat_df['value'],
        x=heat_df['index'],
        y=heat_df['variable'],
        colorscale='purples',
        showlegend=False)
    fig2 = go.Figure([])
    fig2.add_trace(heatmap)
    fig2.update_layout(clickmode='event+select')
    fig2.update_traces(customdata=heat_df[['index','variable']])
    fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    fig2['layout']['yaxis']['autorange'] = "reversed"
    fig2.update_layout(template=template)

    fig1.update_layout(title=title1)
    fig2.update_layout(title=title2)
    return fig1,fig2


@app.callback(
            Output('graph-barchart-network','clickData'),
            Input('dropdown-network','value'),
            Input('dropdown-network-barchart','value')
            )
def update_clickdata(network,feature):
    dff = network_df[network_df['Base']==network]
    bar_data = dff[feature]
    dff1 = ast.literal_eval(bar_data.values[0])
    if feature == 'top_language_use':
        dff1 = {language_dictionary[k]:v for k,v in dff1.items()}
    base_click = list(dff1.keys())[0]
    clickData = {'points': [{'customdata': base_click}]}
    return clickData

## Top Country by Token Barchart - Network Level
@app.callback(
            Output('graph-top-countries-network','figure'),
            Input('dropdown-network','value'),
            Input('graph-barchart-network','clickData'),
            Input('dropdown-network-barchart','value')
            )
def top_country_barcharts(network,clickData,feature):
    if network == 'Russia':
        dff = rus_top_tokens
        demonym = 'Russian'
    elif network == 'United States of America':
        dff = us_top_tokens
        demonym = 'American'
    token = clickData['points'][0]['customdata']
    if feature == 'top_language_use':
        lookup_lang = {v: k for k, v in language_dictionary.items()}
        token = lookup_lang[token]
    toppers = dff.sort_values(by=token,ascending=False)[:10]
    toppers = dict(toppers[token])
    x = list(toppers.keys())
    y = list(toppers.values())
    token_df = pd.DataFrame(x,columns=['key'])
    token_df['value'] = y
    token_df = token_df[::-1]
    fig = go.Figure([])
    title=f"{demonym} Embassies' Use of '{token}'"
    barchart = go.Bar(y=token_df['key'],x=token_df['value'],orientation='h')
    fig.add_trace(barchart)
    fig.update_layout(margin=dict(l=30, r=30, t=40, b=30))
    fig.update_layout(title=title)
    fig.update_layout(template=template)
    return fig

## Timeseries - Network Level - (Makes the multi-input timeseries for network-level data)
@app.callback(
            Output('graph-timeseries-network','figure'),
            [Input('dropdown-network','value'),
            Input('dropdown-network-timeseries','value')]
            )
def update_network_timeseries(network,items):
    dff = network_df[network_df['Base']==network]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for item in items:
        if item == 'sentiment':
            secondary_y=True
        else:
            secondary_y=False
        frequency_dict = ast.literal_eval(dff[item].values[0])
        x = list(frequency_dict.keys())
        y = [np.round(float(idx),3) for idx in list(frequency_dict.values())]
        fig.add_trace(go.Scatter(x=x, y=y,
                        mode='lines',
                        name=item,
                        line_shape='spline',
                        fill='tozeroy'),
                        secondary_y=secondary_y)
    title = f"Time Series Data of {network}'s Diplomatic Network'"
    fig.update_layout(showlegend=True)
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=30))
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(template=template)
    fig.update_layout(height=600)
    fig.update_layout(title=title)
    return fig

### Timeseries - Embassy Level - (Makes the multi-input timeseries for embassy-level data)
@app.callback(
            Output('graph-timeseries-embassy','figure'),
            [Input('dropdown-network','value'),
            Input('dropdown-embassy','value'),
            Input('dropdown-timeseries-embassy','value')]
            )
def update_embassy_timeseries(network,country,items):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for item in items:
        if item == 'sentiment':
            secondary_y=True
            label = 'Sentiment'
        else:
            secondary_y=False
            label = 'Frequency Data'
        frequency_dict = ast.literal_eval(dff[item].values[0])
        x = list(frequency_dict.keys())
        y = [np.round(float(idx),3) for idx in list(frequency_dict.values())]
        fig.add_trace(go.Scatter(x=x, y=y,
                        mode='lines',
                        name=item,
                        line_shape='spline',
                        fill='tozeroy'),
                        secondary_y=secondary_y)
    fig.update_layout(showlegend=True)
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=30))
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(template=template)
    title = f"{label} of {network}'s Embassy in {country}"
    fig.update_layout(title=title)
    return fig

## Barchart and Heatmap - Embassy-Level - (Makes the bar chart and heatmap at the network level)
@app.callback(
            Output('graph-barchart-embassy','figure'),
            Output('graph-heatmap-embassy','figure'),
            Input('dropdown-network','value'),
            Input('dropdown-embassy','value'),
            Input('dropdown-token-embassy','value')
            )
def update_embassy_token_charts(network,country,feature):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    bar_data = dff[feature]
    heat_data = dff[feature+'_timeseries']
    dff1 = ast.literal_eval(bar_data.values[0])
    if feature == 'top_language_use':
        dff1 = {language_dictionary[k]:v for k,v in dff1.items()}
    x = list(dff1.keys())[:20]
    y = list(dff1.values())[:20]
    token_df = pd.DataFrame(x,columns=['key'])
    token_df['value'] = y
    token_df = token_df[::-1]
    fig1 = go.Figure([])
    barchart = go.Bar(y=token_df['key'],x=token_df['value'],orientation='h')
    fig1.add_trace(barchart)
    fig1.update_layout(template=template)
    fig1.update_layout(margin=dict(l=30, r=30, t=30, b=30))
    dff2 = pd.DataFrame.from_dict(ast.literal_eval(heat_data.values[0])).reset_index()
    dff2 = dff2[list(dff2.columns)[:21]]
    list_of_columns = list(dff2.columns)[1:]
    list_of_columns = list_of_columns.reverse()
    heat_df = pd.melt(dff2, id_vars=['index'], value_vars=list_of_columns)
    heat_df = heat_df.replace(0, np.nan)
    heatmap = go.Heatmap(
        z=heat_df['value'],
        x=heat_df['index'],
        y=heat_df['variable'],
        colorscale='greens',
        showlegend=False)
    fig2 = go.Figure([])
    fig2.add_trace(heatmap)
    fig2.update_layout(clickmode='event+select')
    fig2.update_traces(customdata=heat_df[['index','variable']])
    fig2.update_layout(margin=dict(l=30, r=30, t=30, b=30))
    fig2['layout']['yaxis']['autorange'] = "reversed"
    fig2.update_layout(template=template)

    feature_name = label_dictionary[feature]
    title1 = f"{feature_name} of {network}'s Embassy in {country}"
    title2 = f"{feature_name} of {network}'s Embassy in {country} Over Time"

    fig1.update_layout(title=title1)
    fig2.update_layout(title=title2)
    return fig1,fig2

## Word Cloud - Embassy level - (Creates the Wordcloud (Or Sometimes Pie Chart) for embassy level)
@app.callback(
            Output('graph-wordcloud-embassy','figure'),
            Input('dropdown-network','value'),
            Input('dropdown-embassy','value'),
            Input('dropdown-token-embassy','value')
            )
def update_wordcloud(network, country, feature):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    dictionary = ast.literal_eval(dff[feature].values[0])
    not_for_wordcloud = set(['top_language_use','top_emojis'])
    if feature in not_for_wordcloud:
        if feature == 'top_language_use':

            dictionary = {language_dictionary[k]:v for k,v in dictionary.items()}
            main_dict = dict(zip(list(dictionary.keys())[:4],list(dictionary.values())[:4]))
            main_dict['other'] = sum(list(dictionary.values())[4:])
            dff2 = pd.DataFrame.from_dict(main_dict,orient='index').reset_index()
            fig = px.pie(dff2, values=0, names='index', color='index')
        else:
            #dictionary = {language_dictionary[k]:v for k:v in dictionary}
            main_dict = dict(zip(list(dictionary.keys())[:7],list(dictionary.values())[:7]))
            main_dict['other'] = sum(list(dictionary.values())[7:])
            dff2 = pd.DataFrame.from_dict(main_dict,orient='index').reset_index()
            fig = px.pie(dff2, values=0, names='index', color='index')
    else:
        my_wordcloud = WordCloud(
                background_color='#B6BCD5',
                height=400
            ).generate_from_frequencies(dictionary)
        fig = px.imshow(my_wordcloud, template=template)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    #fig.update_layout(plot_bgcolor='#B6BCD5')
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    #fig.update_layout(,paper_bgcolor="#AFB3C4")
    fig.update_layout(template=template)
    return fig

### Info Card - Embassy Level - (Creates the Embassy-level Infocard information )
@app.callback(
            Output('embassy-name-card','children'),
            Output('embassy-screen_name','children'),
            Output('embassy-description','children'),
            Output('embassy-created_at','children'),
            Output('embassy-statuses_count','children'),
            Output('embassy-samples','children'),
            Output('embassy-ratio-retweets','children'),
            Output('embassy-ratio-original','children'),
            Input('dropdown-network','value'),
            Input('dropdown-embassy','value'),
            )
def update_embassy_info(network, country):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]

    name = dff['Country'].values
    screen_name = '@{}'.format(dff['screen_name'].values[0])
    #dff['new_created_at'] = pd.to_datetime(dff['created_at']).dt.date
    location = 'Location: {}'.format((dff['location']).values[0])
    statuses_count = 'Statuses Count: {}'.format(dff['statuses_count'].values[0])
    description = 'Description: {}'.format(dff['description'].values[0])
    tweets = ast.literal_eval(dff.original_tweets.values[0])
    tweets_df = pd.DataFrame(tweets).transpose().rename(columns={0:'full_text',1:'lang',2:'clean_text',3:'entities',4:'retweet_bool'}).reset_index()
    sample_ratio = np.round(len(tweets_df)/dff['statuses_count'].values[0],3)*100
    sample_ratio_text = 'Sample Size: {}% of Tweets'.format(np.round(sample_ratio,3))
    ratio_retweets = np.round(len(tweets_df[tweets_df['retweet_bool']==True])/len(tweets_df),4) * 100
    ratio_retweets_text = 'Ratio Retweets: {}%'.format(np.round(ratio_retweets,3))
    ratio_original = np.round(len(tweets_df[tweets_df['retweet_bool']==False])/len(tweets_df),3) * 100
    ratio_original_text = 'Ratio Original Tweets: {}%'.format(np.round(ratio_original,3))

    return name, screen_name, description, location, statuses_count, sample_ratio_text, ratio_retweets_text, ratio_original_text

### Table Markdown of Twitter Info - (Creates the Embassy-level Original Tweets Table)
@app.callback(
            Output('graph-textmarkdown-embassy','figure'),
            Output('header-h1-embassy','children'),
            Input('graph-heatmap-embassy','clickData'),
            Input('dropdown-network','value'),
            Input('dropdown-embassy','value'),
            Input('dropdown-token-embassy','value')
            )
def update_table(clickData,network,country,feature):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    clicks = clickData['points'][0]['customdata']
    tweets = ast.literal_eval(dff.original_tweets.values[0])
    tweets_df = pd.DataFrame(tweets).transpose().rename(columns={0:'full_text',1:'lang',2:'clean_text',3:'entities',4:'retweet_bool'}).reset_index()
    tweets_df['index'] = pd.to_datetime(tweets_df['index'])
    start_date = pd.to_datetime(clicks[0]) - pd.DateOffset(31)
    end_date = pd.to_datetime(clicks[0])
    token = clicks[1]
    filtered_df = tweets_df[(tweets_df['index'] >= start_date) & (tweets_df['index'] <= end_date)]
    if feature == 'top_language_use':
        counter_df = filtered_df[filtered_df['lang']==token][['index','full_text']]
    else:
        counter_df = filtered_df[filtered_df['full_text'].str.contains(token,flags=re.IGNORECASE)][['index','full_text']]
    fig = go.Figure(data=[go.Table(header=dict(values=['index', 'full_text']),
                  columnorder = [1,2],
                  columnwidth = [30,70],
                  cells=dict(values=[list(counter_df['index'].dt.date), list(counter_df.full_text)]))])
    #fig.update_layout(plot_bgcolor='#B6BCD5')
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=30))
    #fig.update_layout(paper_bgcolor="#AFB3C4")
    title = 'Click on Heatmap to See Original Tweets:'
    section_header = "Twitter Operations of {} in {}".format(network,country)
    fig.update_layout(title=title)
    fig.update_layout(template=template)
    return fig, section_header

### DROPDOWNS - Here we have all the dropdowns used to create the comparative timeseries
### Compare Dropdown - Embassy Level
@app.callback(Output('dropdown-compare-embassy-1','options'),
            Output('dropdown-compare-embassy-1','value'),
            Input('dropdown-compare-network-1','value'))
def update_dropdowns(network):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['HANDLE']!='0')]
    options_list = list(dff['Country'].values)
    options = [{'label': i, 'value': i} for i in options_list]
    value = options[0]['label']
    return options, value
@app.callback(Output('dropdown-compare-embassy-2','options'),
            Output('dropdown-compare-embassy-2','value'),
            Input('dropdown-compare-network-2','value'))
def update_dropdowns(network):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['HANDLE']!='0')]
    options_list = list(dff['Country'].values)
    options = [{'label': i, 'value': i} for i in options_list]
    value = options[0]['label']
    return options, value
### Compare Dropdown - Indicator Level
@app.callback(Output('dropdown-compare-indicator-1','options'),
            Output('dropdown-compare-indicator-1','value'),
            Input('dropdown-compare-embassy-1','value'),
            Input('dropdown-compare-network-1','value'))
def update_dropdowns(country,network):
    dff2 = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    dff3 = pd.DataFrame(dff2.iloc[0])
    dff3 = dff3.reset_index()
    columns = list(dff3.columns)
    available_columns = dff3[dff3[columns[1]]!='{}'][dff3['index'].str.contains('timeseries')==False]['index'].values
    available_columns = [x for x in list(label_dictionary.keys()) if x in set(available_columns)]
    avc = []
    for col in available_columns:
        timeseries = ast.literal_eval(dff2[col+'_timeseries'].iloc[0])
        pops = []
        for key in timeseries:
             if len(timeseries[key])<2:
                 pops.append(key)
        for pop in pops:
             del timeseries[pop]
        if len(timeseries) > 1:
             avc.append(col)
    options = [{'label': label_dictionary[i],'value': i} for i in list(avc)]
    options.append({'label': 'Frequency Data','value': 'frequency_data'})
    value = options[-1]['value']
    return options, value
@app.callback(Output('dropdown-compare-indicator-2','options'),
            Output('dropdown-compare-indicator-2','value'),
            Input('dropdown-compare-embassy-2','value'),
            Input('dropdown-compare-network-2','value'))
def update_dropdowns(country,network):
    dff2 = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    dff3 = pd.DataFrame(dff2.iloc[0])
    dff3 = dff3.reset_index()
    columns = list(dff3.columns)
    available_columns = dff3[dff3[columns[1]]!='{}'][dff3['index'].str.contains('timeseries')==False]['index'].values
    available_columns = [x for x in list(label_dictionary.keys()) if x in set(available_columns)]
    avc = []
    for col in available_columns:
        timeseries = ast.literal_eval(dff2[col+'_timeseries'].iloc[0])
        pops = []
        for key in timeseries:
             if len(timeseries[key])<2:
                 pops.append(key)
        for pop in pops:
             del timeseries[pop]
        if len(timeseries) > 1:
             avc.append(col)
    options = [{'label': label_dictionary[i],'value': i} for i in list(avc)]
    options.append({'label': 'Frequency Data','value': 'frequency_data'})
    value = options[-1]['value']

    return options, value
## Compare Dropdown - Token Level
@app.callback(Output('dropdown-compare-token-1','options'),
            Output('dropdown-compare-token-1','value'),
            Input('dropdown-compare-embassy-1','value'),
            Input('dropdown-compare-network-1','value'),
            Input('dropdown-compare-indicator-1','value'))
def update_dropdowns(country,network,feature):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    if feature == 'frequency_data':
        frequency_options = {'frequency_tweets':'Frequency of Tweets', 'frequency_retweets':'Frequency of Retweets', 'favorites_counts':'Favorite Counts', 'retweets_counts': 'Retweeted Counts','sentiment':'Vader Sentiment Analysis'}
        options = [{'label': v, 'value': k} for k,v in frequency_options.items()]
    else:
        bar_data = dff[feature+'_timeseries']
        d = ast.literal_eval(bar_data.values[0])
        pops = []
        for key in d:
            if len(d[key]) < 2:
                pops.append(key)
        for pop in pops:
            del d[pop]
        options_list = list(d.keys())
        options = [{'label': i, 'value': i} for i in options_list]
    value = options[0]['value']
    return options, value

@app.callback(Output('dropdown-compare-token-2','options'),
            Output('dropdown-compare-token-2','value'),
            Input('dropdown-compare-embassy-2','value'),
            Input('dropdown-compare-network-2','value'),
            Input('dropdown-compare-indicator-2','value'))
def update_dropdowns(country,network,feature):
    dff = embassy_df[(embassy_df['Base']==network)&(embassy_df['Country']==country)]
    if feature == 'frequency_data':
        frequency_options = {'frequency_tweets':'Frequency of Tweets', 'frequency_retweets':'Frequency of Retweets', 'favorites_counts':'Favorite Counts', 'retweets_counts': 'Retweeted Counts','sentiment':'Vader Sentiment Analysis'}
        options = [{'label': v, 'value': k} for k,v in frequency_options.items()]
    else:
        bar_data = dff[feature+'_timeseries']
        d = ast.literal_eval(bar_data.values[0])
        pops = []
        for key in d:
            if len(d[key]) < 2:
                pops.append(key)
        for pop in pops:
            del d[pop]
        options_list = list(d.keys())
        options = [{'label': i, 'value': i} for i in options_list]

    value = options[0]['value']
    return options, value



## Comparative Timeseries - (Creates the timeseries with 'historical event bars')
@app.callback(Output('graph-timeseries-compare','figure'),
            Output('dtw-score','children'),
            Output('correlation','children'),
            Input('dropdown-compare-token-2','value'),
            Input('dropdown-compare-embassy-2','value'),
            Input('dropdown-compare-network-2','value'),
            Input('dropdown-compare-indicator-2','value'),
            Input('dropdown-compare-token-1','value'),
            Input('dropdown-compare-embassy-1','value'),
            Input('dropdown-compare-network-1','value'),
            Input('dropdown-compare-indicator-1','value'),
            Input('historical-event-dropdown','value'))
def update_timeline(items1,country1,network1,token_type1,items2,country2,network2,token_type2,event):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig = go.Figure()
    dff1 = embassy_df[(embassy_df['Base']==network1)&(embassy_df['Country']==country1)]
    dff2 = embassy_df[(embassy_df['Base']==network2)&(embassy_df['Country']==country2)]
    if items1 == 'sentiment' or items2 == 'sentiment':
        secondary_y=True
    else:
        secondary_y=False

    if token_type1 != 'frequency_data':
        token_dictionary = ast.literal_eval(dff1[token_type1+'_timeseries'].values[0])
        timeline_data1 = token_dictionary[items1]
        x1 = list(timeline_data1.keys())
        y1 = [np.round(float(idx),3) for idx in list(timeline_data1.values())]
        fig.add_trace(go.Scatter(x=x1, y=y1,
                        mode='lines',
                        name=items1 + "-of-" + network1 + "-in-" + country1,
                        line_shape='spline',
                        fill='tozeroy'),
                        secondary_y=secondary_y)
    else:
        timeline_data1 = ast.literal_eval(dff1[items1].values[0])
        x1 = list(timeline_data1.keys())
        y1 = [np.round(float(idx),3) for idx in list(timeline_data1.values())]
        fig.add_trace(go.Scatter(x=x1, y=y1,
                        mode='lines',
                        name=items1 + "-of-" + network1 + "-in-" + country1,
                        line_shape='spline',
                        fill='tozeroy'),
                        secondary_y=secondary_y)
    if token_type2 != 'frequency_data':
        token_dictionary = ast.literal_eval(dff2[token_type2+'_timeseries'].values[0])
        timeline_data2 = token_dictionary[items2]
        x2 = list(timeline_data2.keys())
        y2 = [np.round(float(idx),3) for idx in list(timeline_data2.values())]
        fig.add_trace(go.Scatter(x=x2, y=y2,
                            mode='lines',
                            name=items2 + "-of-" + network2 + "-in-" + country2,
                            line_shape='spline',
                            fill='tozeroy'),
                            secondary_y=secondary_y)
    else:
        timeline_data2 = ast.literal_eval(dff2[items2].values[0])
        x2 = list(timeline_data2.keys())
        y2 = [np.round(float(idx),3) for idx in list(timeline_data2.values())]
        fig.add_trace(go.Scatter(x=x2, y=y2,
                        mode='lines',
                        name=items2 + "-of-" + network2 + "-in-" + country2,
                        line_shape='spline',
                        fill='tozeroy'),
                        secondary_y=secondary_y)

    if event == '':
        pass
    else:
        events_dff = events_df[events_df['event']==event]
        x_s = events_dff['start'].values[0]
        x_e = events_dff['end'].values[0]
        text = events_dff['event'].values[0]
        max_date = max(pd.to_datetime(x1+x2))
        min_date = min(pd.to_datetime(x1+x2))
        height = max(y1+y2)-(max(y1+y2)*.20)
        fig.add_vrect(x0=x_s,
                    x1=x_e,
                    fillcolor="green",
                    opacity=0.6,
                    layer="below",
                    line_width=0)
        fig.add_annotation(
            x=x_s,
            y=height,
            xref="x",
            yref="y",
            text=text,
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
                ),
            align="right",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="white",
            xanchor='right',
            ax=20,
            ay=-50,
            bordercolor="white",
            borderwidth=2,
            borderpad=4,
            bgcolor="green",
            opacity=0.8
            )

    fig.update_layout(showlegend=True)
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=30))
    fig.update_layout(height=800)
    fig.update_layout(template=template)

### Dynamic Time Warping Score
    timeline_data1 = {k:np.round(float(v),3) for k,v in timeline_data1.items()}
    timeline_data2 = {k:np.round(float(v),3) for k,v in timeline_data2.items()}
    timeline_data1 = pd.DataFrame.from_dict(timeline_data1, orient='index').fillna(0).reset_index()
    timeline_data2 = pd.DataFrame.from_dict(timeline_data2, orient='index').fillna(0).reset_index()

    s3 = timeline_data1.merge(timeline_data2, how='inner', on='index')
    if len(s3) > 2:
        columns = list(s3.columns)
        dtw_score = dtw(s3[columns[-2]], s3[columns[-1]])
        dtw_score = np.round(dtw_score,3)
        pearsonr = stats.pearsonr(s3[columns[-2]], s3[columns[-1]])
        correlation_coefficient = np.round(pearsonr[0],4)
        p_value = np.format_float_scientific(pearsonr[1], precision=3)
    else:
        dtw_score = "N/A"
        pearsonr = "N/A"
        correlation_coefficient = "N/A"
        p_value = "N/A"

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig, dtw_score, correlation_coefficient

### Map - (Creates the Network Map)
@app.callback(Output('graph-big-map', 'figure'),
              Input('dropdown-network', 'value'),
              Input('dropdown-bubble', 'value'))
def display_choropleth(network, dropdown_bubble):
    dff = embassy_df[embassy_df['Base']==network]
    facts = ['TRUE','SELF']
    dff = dff[dff['Embassy'].isin(facts)].fillna(0)
    dff['statuses_count'] = dff['statuses_count'].fillna(False)

    embassy_connections = pd.DataFrame(dff.groupby(['Source', 'ISO_A3']).size().reset_index(name='embassy_connections'))
    embassy_keys = dff['ISO_A3'].unique().tolist() #dff['Source'].append(dff['ISO_A3']).unique().tolist()
    edge_df = embassy_connections
    node_df = dff
    nodes = embassy_keys
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for index, row in edge_df.iterrows():
        G.add_edge(row['Source'], row['ISO_A3'], weight=row['embassy_connections'])
        x, y = node_df['CapitalLongitude'].values, node_df['CapitalLatitude'].values
    pos_dict = {}
    for index, iata in enumerate(node_df['ISO_A3']):
        pos_dict[iata] = (x[index], y[index])
    for iata, coordinate in pos_dict.items():
        G.nodes[iata]['pos'] = coordinate
    fig = go.Figure()

    #Add edges
    for i in range(len(edge_df)):
        fig.add_trace(
            go.Scattermapbox(
                lon=[pos_dict[edge_df['Source'][i]][0], pos_dict[edge_df['ISO_A3'][i]][0]],
                lat=[pos_dict[edge_df['Source'][i]][1], pos_dict[edge_df['ISO_A3'][i]][1]],
                mode='lines',
                line=dict(width=0.5, color=np.where(edge_df['Source'][i] == 'RUS', 'red', 'blue').tolist()),
                opacity=0.8,
                hoverinfo='none',
                )
            )
    if dropdown_bubble == 'statuses_count':
        column_key = 'statuses_count'
        scaler = MinMaxScaler()
        size_scaler1 = scaler.fit_transform(np.array(dff[column_key]).reshape(-1, 1)) * 2000
        size_scaler2 = scaler.fit_transform(np.array(dff[column_key]).reshape(-1, 1)) * 1000
        colors_1 = 'blues'
        colors_2 = 'reds'
    elif dropdown_bubble == 'followers_count':
        column_key = 'followers_count'
        scaler = MinMaxScaler()
        size_scaler1 = scaler.fit_transform(np.array(dff[column_key]).reshape(-1, 1)) * 2000
        size_scaler2 = scaler.fit_transform(np.array(dff[column_key]).reshape(-1, 1)) * 1000
        colors_1 = 'greens'
        colors_2 = 'blues'
    else:
        None

    fig.add_trace(go.Scattermapbox(
        lat=node_df['CapitalLatitude'],
        lon=node_df['CapitalLongitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=size_scaler1,
            sizeref=1.1,
            sizemode="area",
            opacity=1,
            color=dff[column_key],
            #color='grey',
            colorscale=colors_1
        ),
        hoverinfo='skip',
        text=dff['ISO_A3']
        ))

    fig.add_trace(go.Scattermapbox(
        lat=node_df['CapitalLatitude'],
        lon=node_df['CapitalLongitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=size_scaler2,
            sizeref=1.1,
            sizemode="area",
            opacity=1,
            color=dff[column_key],
            colorscale=colors_2

        ),
        hoverinfo='text',
        hovertext=dff['Country'],
        text=dff['ISO_A3']
        ))
    fig.update_layout(showlegend=False)
    fig.update_traces(customdata=node_df['Country'])
    fig.update_layout(
            margin={"r":20,"t":40,"l":20,"b":20},
            mapbox = {
                    'accesstoken': token,
                    'style': "carto-darkmatter",
                    'center':{'lat':35,'lon':14},
                    'zoom': 2},
            showlegend = False)
    fig.update_layout(template=template)
    fig.update_layout(height=600)

    title = f"Map of {network}'s Diplomatic Network"
    fig.update_layout(title=title)
    return fig



### Popup Callbacks
@app.callback(
    Output("popover-embassy-info", "is_open"),
    [Input("popover-embassy-button", "n_clicks")],
    [State("popover-embassy-info", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover-embassy-info-2", "is_open"),
    [Input("popover-embassy-button-2", "n_clicks")],
    [State("popover-embassy-info-2", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover-network-info", "is_open"),
    [Input("popover-network-button", "n_clicks")],
    [State("popover-network-info", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover-comparative-info", "is_open"),
    [Input("popover-comparative-button", "n_clicks")],
    [State("popover-comparative-info", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open





# # Activate this callback to visualize clickback data
# @app.callback(Output('output', 'children'), [
#     Input('graph-barchart-network', 'clickData'),
#     ])
# def update_output(args):
#     return json.dumps(args, indent=2)
