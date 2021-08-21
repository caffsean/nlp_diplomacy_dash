"""
Created - July 2021
Author - caffsean

This file is a MODULE containing functions used to preprocess the database data to make it suitable for the dashboard
"""


import ast
import pandas as pd
from collections import defaultdict
import re
from emoji import UNICODE_EMOJI
import demoji
import operator
import numpy as np
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
pd.options.mode.chained_assignment = None

import db_call


## Preprocess Emojis
def process_emojis(df,n=20):
    """
    input - df
            n - the number of top emojis
    output - emoji count dictionary, emoji timeseries dictionaries
    """
    d = defaultdict(int)
    for idx in range(len(df)):
        tokens = list(demoji.findall(df.full_text.iloc[idx]).keys())
        for t in tokens:
            d[t] += 1
    d = dict(sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    top_d = dict(zip(list(d.keys())[:n],list(d.values())[:n]))
    token_list = list(top_d.keys())
    full_time_dict = {}
    for token in token_list:
        dff = df[df['full_text'].str.contains(token)][['created_at','full_text']]
        if len(dff) > 0:
            dff['count'] = 1
            dff['created_at'] = pd.to_datetime(dff['created_at'])
            dff = dff.resample('M', on='created_at').sum().reset_index().sort_values('created_at')
            dff['token'] = token
            one_token_dict = dict(zip(list(dff.created_at.astype(str)),list(dff['count'])))
            full_time_dict[token] = one_token_dict
    return top_d, full_time_dict

## Preprocess Hashtags
def process_hashtags(user_name,n=20):
    hash_df = db_call.get_hashtags_from_user(user_name)
    d = dict(hash_df.hashtag.value_counts())
    top_d = dict(zip(list(d.keys())[:n],list(d.values())[:n]))
    token_list = list(top_d.keys())
    full_time_dict = {}
    for token in token_list:
        dff = hash_df[hash_df['hashtag']==token]
        dff['count'] = 1
        dff['created_at'] = pd.to_datetime(dff['created_at'])
        dff = dff.resample('M', on='created_at').sum().reset_index().sort_values('created_at')
        dff['token'] = token
        one_token_dict = dict(zip(list(dff.created_at.astype(str)),list(dff['count'])))
        full_time_dict[token] = one_token_dict
    return top_d, full_time_dict

## Preprocess Hashtags - MULTIPLE
def process_hashtags_multiple(user_names_list,n=20):
    hash_df = db_call.get_hashtags_from_user(user_names_list[0])
    for handle in tqdm(user_names_list[1:]):
        df2 = db_call.get_hashtags_from_user(handle)
        hash_df = pd.concat([hash_df,df2])
    d = dict(hash_df.hashtag.value_counts())
    top_d = dict(zip(list(d.keys())[:n],list(d.values())[:n]))
    token_list = list(top_d.keys())
    full_time_dict = {}
    for token in token_list:
        dff = hash_df[hash_df['hashtag']==token]
        dff['count'] = 1
        dff['created_at'] = pd.to_datetime(dff['created_at'])
        dff = dff.resample('M', on='created_at').sum().reset_index().sort_values('created_at')
        dff['token'] = token
        one_token_dict = dict(zip(list(dff.created_at.astype(str)),list(dff['count'])))
        full_time_dict[token] = one_token_dict
    return top_d, full_time_dict

### Preprocess User Mentions

def process_user_mentions(user_name,n=20):
    ment_df = db_call.get_mentions_from_user(user_name)
    d = dict(ment_df.screen_name.value_counts())
    top_d = dict(zip(list(d.keys())[:n],list(d.values())[:n]))
    token_list = list(top_d.keys())
    full_time_dict = {}
    for token in token_list:
        dff = ment_df[ment_df['screen_name']==token]
        dff['count'] = 1
        dff['created_at'] = pd.to_datetime(dff['created_at'])
        dff = dff.resample('M', on='created_at').sum().reset_index().sort_values('created_at')
        dff['token'] = token
        one_token_dict = dict(zip(list(dff.created_at.astype(str)),list(dff['count'])))
        full_time_dict[token] = one_token_dict
    return top_d, full_time_dict


### Preprocess User Mentions - MULTIPLE
def process_user_mentions_multiple(user_names_list,n=20):
    ment_df = db_call.get_mentions_from_user(user_names_list[0])
    for handle in tqdm(user_names_list[1:]):
        df2 = db_call.get_mentions_from_user(handle)
        ment_df = pd.concat([ment_df,df2])
    d = dict(ment_df.screen_name.value_counts())
    top_d = dict(zip(list(d.keys())[:n],list(d.values())[:n]))
    token_list = list(top_d.keys())
    full_time_dict = {}
    for token in token_list:
        dff = ment_df[ment_df['screen_name']==token]
        dff['count'] = 1
        dff['created_at'] = pd.to_datetime(dff['created_at'])
        dff = dff.resample('M', on='created_at').sum().reset_index().sort_values('created_at')
        dff['token'] = token
        one_token_dict = dict(zip(list(dff.created_at.astype(str)),list(dff['count'])))
        full_time_dict[token] = one_token_dict
    return top_d, full_time_dict


### Process Frequencies
def process_tweet_frequency(embassy_key, embassy_df):
    """
    ## Time period is set to 'Month'
    input - df
    output - dictionary of tweet frequency
            dictionary of retweet frequency
            dictionary of favorite count
            dictionary of retweet count
    """
    counter_df = embassy_df
    counter_df['count'] = 1
    user_id = db_call.get_user_id(embassy_key)
    counter_df['retweet_bool'] = [True if str(embassy_df.iloc[idx].user_id)!=user_id else False for idx in range(len(embassy_df))]
    counter_df['created_at'] = pd.to_datetime(embassy_df['created_at'])
    counter_df = counter_df.resample('M', on='created_at').sum().reset_index().sort_values('created_at')

    frequency_tweets = dict(zip(list(counter_df['created_at'].astype(str)),list(counter_df['count'])))
    frequency_is_retweet = dict(zip(list(counter_df['created_at'].astype(str)),list(counter_df['retweet_bool'])))
    frequency_favorites_count = dict(zip(list(counter_df['created_at'].astype(str)),list(counter_df['retweet_count'])))
    frequency_retweets_count = dict(zip(list(counter_df['created_at'].astype(str)),list(counter_df['favorite_count'])))

    return frequency_tweets, frequency_is_retweet, frequency_favorites_count,frequency_retweets_count


## Process Frequencies
def process_tweet_frequency_multiple(embassy_key, embassy_df):
    """
    ## Time period is set to 'Month'
    input - df
    output - dictionary of tweet frequency
            dictionary of retweet frequency
            dictionary of favorite count
            dictionary of retweet count
    """
    counter_df = embassy_df
    counter_df['count'] = 1
    user_id = db_call.get_user_id_multiple(embassy_key)
    counter_df['retweet_bool'] = [True if str(embassy_df.iloc[idx].user_id)!=user_id else False for idx in range(len(embassy_df))]
    counter_df['created_at'] = pd.to_datetime(embassy_df['created_at'])
    counter_df = counter_df.resample('M', on='created_at').sum().reset_index().sort_values('created_at')

    frequency_tweets = dict(zip(list(counter_df['created_at'].astype(str)),list(counter_df['count'])))
    frequency_is_retweet = dict(zip(list(counter_df['created_at'].astype(str)),list(counter_df['retweet_bool'])))
    frequency_favorites_count = dict(zip(list(counter_df['created_at'].astype(str)),list(counter_df['retweet_count'])))
    frequency_retweets_count = dict(zip(list(counter_df['created_at'].astype(str)),list(counter_df['favorite_count'])))

    return frequency_tweets, frequency_is_retweet, frequency_favorites_count,frequency_retweets_count

## Process Language
def process_language(df):
    '''
        input
            -df
        returns
            -dict of language count
            -dict of dicts of language use for timeseries
    '''
    lang_count = dict(df.lang.value_counts())
    lang_time_series_dict = {}
    for lang in list(lang_count.keys()):
        dff = df[df.lang==lang][['created_at','lang']]
        dff['count'] = 1
        dff['created_at'] = pd.to_datetime(dff['created_at'])
        #dff = dff.set_index('created_at')
        dff = dff.groupby([pd.Grouper(key='created_at', freq='M')])['count'].sum().reset_index().sort_values('created_at')
        dff['token'] = lang
        one_token_dict = dict(zip(list(dff.created_at.astype(str)),list(dff['count'])))
        lang_time_series_dict[lang] = one_token_dict

    return lang_count,lang_time_series_dict

## Process POS
def process_pos(df,language='en',pos='ADJ',n=20):
    """
    input - df
            - language ('en')
            - pos ('ADJ')
            - top n to return
    returns - top part of speech dict
            - timeline dictionaries

    """
    d = defaultdict(int)
    df = df[df.lang == language]
    this_pos = []
    for idx in range(len(df)):
        lemma_list = []
        if (df.iloc[idx].lemmas != None) & (df.iloc[idx].pos != None):
            pos_dict = dict(zip(df.iloc[idx].lemmas.strip('][').split(','),df.iloc[idx].pos.strip('][').split(',')))
            for key in list(pos_dict.keys()):
                if pos_dict[key] == pos:
                    d[key] += 1
                    lemma_list.append(key)
        this_pos.append(lemma_list)

    df['this_pos'] = this_pos
    d = dict(sorted(d.items(), key=operator.itemgetter(1),reverse=True))

    top_d = dict(zip(list(d.keys())[:n],list(d.values())[:n]))

    token_list = list(top_d.keys())
    full_time_dict = {}
    #df['lemmas'] = [lemmas.strip('][').split(',') if lemmas!=None else None for lemmas in df['lemmas']]

    for token in token_list:
        df['count'] = [lemmas.count(token) if lemmas!=None else None for lemmas in df['this_pos']]
        dff = df[df['count'] > 0]
        dff['created_at'] = pd.to_datetime(dff['created_at'])
        dff = dff.groupby([pd.Grouper(key='created_at', freq='M')])['count'].sum().reset_index().sort_values('created_at')
        dff['token'] = token
        one_token_dict = dict(zip(list(dff.created_at.astype(str)),list(dff['count'])))
        full_time_dict[token] = one_token_dict

    return top_d,full_time_dict


### Process Entities

def process_entities(df,language='en',entity_type='PERSON',n=20):
    """
    input   - df
            - language ('en')
            - entity type ('LOC','ORG','PERSON','PER'(Russia),'GPE','FAC','EVENT','NORP'
            - top n to return
    returns - top entities
            - top entities timeseries dictionaries
    """
    d = defaultdict(int)
    df = df[df.lang == language]
    this_entity = []
    for idx in range(len(df)):
        entity_list = []
        if (df.iloc[idx].entities != None) & (df.iloc[idx].entities_labels != None):
            entities_dict = dict(zip(df.iloc[idx].entities.strip('][').split(','),df.iloc[idx].entities_labels.strip('][').split(',')))
            for key in list(entities_dict.keys()):
                if entities_dict[key] == entity_type:
                    d[key] += 1
                    entity_list.append(key)
        this_entity.append(entity_list)
    df['this_entity'] = this_entity
    d = dict(sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    top_d = dict(zip(list(d.keys())[:n],list(d.values())[:n]))
    token_list = list(top_d.keys())
    full_time_dict = {}
    #df['entities'] = [lemmas.strip('][').split(',') if lemmas!=None else None for lemmas in df['entities']]

    for token in token_list:
        df['count'] = [entities.count(token) if entities!=None else None for entities in df['this_entity']]
        dff = df[df['count'] > 0]
        dff['created_at'] = pd.to_datetime(dff['created_at'])
        dff = dff.groupby([pd.Grouper(key='created_at', freq='M')])['count'].sum().reset_index().sort_values('created_at')
        dff['token'] = token
        one_token_dict = dict(zip(list(dff.created_at.astype(str)),list(dff['count'])))
        full_time_dict[token] = one_token_dict

    return top_d,full_time_dict

## Process Sentiment
def process_sentiment(df):

    sid = SentimentIntensityAnalyzer()
    scores = []
    df['clean_text'] = df['clean_text'].fillna('None')
    for tweet in df['clean_text']:
        scores.append(sid.polarity_scores(tweet)['compound'])
        #scores.append(sid.polarity_scores(tweet))

    df['score'] = scores
    df = df.set_index('created_at')
    df = pd.DataFrame(df.resample('M')["score"].mean())
    df = df.reset_index()
    x = df['created_at'].astype(str)
    y = df['score'].astype(str)

    return dict(zip(list(x),list(y)))

### Get Original Tweets (For Table)

def get_original_tweets(embassy_key, embassy_df):
    user_id = db_call.get_user_id(embassy_key)
    embassy_df['retweeted_bool'] = [True if str(embassy_df.iloc[idx].user_id)!=user_id else False for idx in range(len(embassy_df))]
    tweet_dict = {}
    for idx in range(len(embassy_df)):
        date = str(embassy_df.created_at.iloc[idx])
        tweet_dict[date] = [embassy_df.full_text.iloc[idx],embassy_df.lang.iloc[idx],embassy_df.clean_text.iloc[idx],embassy_df.entities.iloc[idx],embassy_df.retweeted_bool.iloc[idx]]
    return tweet_dict
