
"""
Created - July 2021
Author - caffsean

This file contains two functions that output two forms of preprocessed data:
    A NETWORK-level dataframe (in csv form) which contains properties of diplomatic networks Twitter accountS (e.g. ALL Russian Embassy accounts)
    An EMBASSY-level dataframe (in csv form) which contains properties of EACH embassy's Twitter account (e.g. Russian Embassy in Afghanistan account)

These functions require as input a dataframe that includes a column of Twitter handles (e.g. @RusEmbassyKabul). This dataframe is merged with the new data to create a dataframe for the dashboard.
"""

import pandas as pd
import numpy as np
import ast

import db_call
import preprocess_tweets

from tqdm import tqdm

#

#base = pd.read_csv('Preprocess_Dashboard/base_info_handles_SAMPLE.csv')
base = pd.read_csv('Preprocess_Dashboard/base_info_handles.csv')

base['HANDLE'] = base['HANDLE'].fillna(0)
base1 = base[:100]
base2 = base[100:200]
base3 = base[200:300]
base4 = base[300:400]
base5 = base[400:]

list_of_rus_handles = [handle[1:] for handle in list(base[(base['Source']=='RUS') & (base['HANDLE']!=0)]['HANDLE'].values)]
list_of_usa_handles = [handle[1:] for handle in list(base[(base['Source']=='USA') & (base['HANDLE']!=0)]['HANDLE'].values)]

### Populate A Dataframe for Network-Level Twitter Data
def populate_network_dictionary(base_df):
    super_dict = {}
    list_of_embassies_USA = [screen_name[1:] for screen_name in base_df[(base_df['Source']=='USA')& (base['HANDLE']!=0)]['HANDLE'].dropna()]
    list_of_embassies_RUS = [screen_name[1:] for screen_name in base_df[(base_df['Source']=='RUS')& (base['HANDLE']!=0)]['HANDLE'].dropna()]
    list_of_lists = [list_of_embassies_USA,list_of_embassies_RUS]
    embassy_networks = ['USA','RUS']
    for network, embassy_key in zip(embassy_networks,list_of_lists):
            print(f'>>> Initializing {network} network...')

            embassy_df = db_call.get_all_user_activity(embassy_key[0])
            for handle in tqdm(embassy_key[1:]):
                print(f'...getting data from {handle}')
                df2 = db_call.get_all_user_activity(handle)
                embassy_df = pd.concat([embassy_df,df2])

            super_dict[network] = {}

            print("...processing emojis...")
            super_dict[network]['top_emojis'],super_dict[network]['top_emojis_timeseries'] = preprocess_tweets.process_emojis(embassy_df,n=20)
            print("...processing hashtags...")
            super_dict[network]['top_hashtags'],super_dict[network]['top_hashtags_timeseries'] = preprocess_tweets.process_hashtags_multiple(embassy_key,n=20)
            print("...processing user mentions...")
            super_dict[network]['top_user_mentions'],super_dict[network]['top_user_mentions_timeseries'] = preprocess_tweets.process_user_mentions_multiple(embassy_key,n=20)
            print("...processing frequencies...")
            super_dict[network]['frequency_tweets'], super_dict[network]['frequency_retweets'], super_dict[network]['favorites_counts'],super_dict[network]['retweets_counts'] = preprocess_tweets.process_tweet_frequency_multiple(embassy_key, embassy_df)
            print("...processing language usage...")
            super_dict[network]['top_language_use'],super_dict[network]['top_language_use_timeseries'] = preprocess_tweets.process_language(embassy_df)
            print("...processing sentiment...")
            super_dict[network]['sentiment'] = preprocess_tweets.process_sentiment(embassy_df)

            languages = ['English','Russian','French','Spanish']
            language_keys = ['en','ru','fr','es']
            parts_of_speech = ['adjectives','verbs','nouns','proper nouns','adverbs']
            pos_keys = ['ADJ','VERB','NOUN','PROPN','ADV']

            for lang, lang_key in zip(languages,language_keys):
                for pos, pos_key in zip(parts_of_speech,pos_keys):
                    print(f'...processing {lang} {pos}...')
                    super_dict[network][f'spacy_pos_{lang_key}_{pos_key}'],super_dict[network][f'spacy_pos_{lang_key}_{pos_key}_timeseries'] = preprocess_tweets.process_pos(embassy_df,language=lang_key,pos=pos_key,n=20)

            entity_keys = ['PERSON', 'NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','LAW','LANGUAGE','DATE',] ## Not included: PERCENT, TIME, MONEY
            entities = ['People','Nationalities or religious or political groups','Facilities and infrastructure','Companies, agencies, institutions','Geopolitical entities (countries, cities, states)','Non-GPE locations, mountain ranges, bodies of water','Products, objects','Events','Named documents made into laws','Any named language','Mentioned dates',]


            for lang, lang_key in zip(languages[:2], language_keys[:2]):
                for ent, ent_key in zip(entities, entity_keys):
                    if (lang_key=='ru') &  (ent_key=='PERSON'):
                          ent_key = 'PERS'
                    print(f'...processing {lang} entities -- ({ent_key}) -- {ent}...')
                    super_dict[network][f'spacy_ent_{lang_key}_{ent_key}'],super_dict[network][f'spacy_ent_{lang_key}_{ent_key}_timeseries'] = preprocess_tweets.process_entities(embassy_df, language=lang_key, entity_type=ent_key, n=20)

    final_df = pd.DataFrame.from_dict(super_dict,orient='index')
    return final_df.to_csv('network_db_SAMPLE.csv')


### Populate A Dataframe for Embassy-Level Twitter Data
def populate_embassy_dictionary(base_df, filename):
    super_dict = {}
    handles = [handle[1:] for handle in base_df[base_df['HANDLE']!=0]['HANDLE']]
    handles.remove('StateDept')
    #handles.remove('mfa_russia')
    for handle in tqdm(handles):
        embassy_key = handle
        print(f">>>INITIALIZING PREPROCESS: {embassy_key}...")
        embassy_df = db_call.get_all_user_activity(embassy_key)
        info_df = db_call.get_all_user_info(embassy_key)
        stats_df = db_call.get_all_user_stats(embassy_key)

        super_dict[embassy_key] = {}

        print("...processing profile data...")
        super_dict[embassy_key]['screen_name'] = info_df.screen_name.iloc[-1]
        super_dict[embassy_key]['name'] = info_df.name.iloc[-1]
        super_dict[embassy_key]['created_at'] = info_df.created_at.iloc[-1]
        super_dict[embassy_key]['description'] = info_df.description.iloc[-1]
        super_dict[embassy_key]['location'] = info_df.location.iloc[-1]
        super_dict[embassy_key]['statuses_count'] = stats_df.statuses_count.iloc[-1]
        super_dict[embassy_key]['followers_count'] = stats_df.followers_count.iloc[-1]
        print("...processing emojis...")
        super_dict[embassy_key]['top_emojis'],super_dict[embassy_key]['top_emojis_timeseries'] = preprocess_tweets.process_emojis(embassy_df,n=20)
        print("...processing hashtags...")
        super_dict[embassy_key]['top_hashtags'],super_dict[embassy_key]['top_hashtags_timeseries'] = preprocess_tweets.process_hashtags(embassy_key,n=20)
        print("...processing user mentions...")
        super_dict[embassy_key]['top_user_mentions'],super_dict[embassy_key]['top_user_mentions_timeseries'] = preprocess_tweets.process_user_mentions(embassy_key,n=20)
        print("...processing frequencies...")
        super_dict[embassy_key]['frequency_tweets'], super_dict[embassy_key]['frequency_retweets'], super_dict[embassy_key]['favorites_counts'],super_dict[embassy_key]['retweets_counts'] = preprocess_tweets.process_tweet_frequency(embassy_key, embassy_df)
        print("...processing language usage...")
        super_dict[embassy_key]['top_language_use'],super_dict[embassy_key]['top_language_use_timeseries'] = preprocess_tweets.process_language(embassy_df)
        print("...processing sentiment...")
        super_dict[embassy_key]['sentiment'] = preprocess_tweets.process_sentiment(embassy_df)
        print("...processing original tweets")
        super_dict[embassy_key]['original_tweets'] = preprocess_tweets.get_original_tweets(embassy_key, embassy_df)

        languages = ['English','Russian','French','Spanish']
        language_keys = ['en','ru','fr','es']
        parts_of_speech = ['adjectives','verbs','nouns','proper nouns','adverbs']
        pos_keys = ['ADJ','VERB','NOUN','PROPN','ADV']

        for lang, lang_key in zip(languages,language_keys):
            for pos, pos_key in zip(parts_of_speech,pos_keys):
                print(f'...processing {lang} {pos}...')
                super_dict[embassy_key][f'spacy_pos_{lang_key}_{pos_key}'],super_dict[embassy_key][f'spacy_pos_{lang_key}_{pos_key}_timeseries'] = preprocess_tweets.process_pos(embassy_df,language=lang_key,pos=pos_key,n=50)

            entity_keys = ['PERSON', 'NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','LAW','LANGUAGE','DATE',] ## Not included: PERCENT, TIME, MONEY
            entities = ['People','Nationalities or religious or political groups','Facilities and infrastructure','Companies, agencies, institutions','Geopolitical entities (countries, cities, states)','Non-GPE locations, mountain ranges, bodies of water','Products, objects','Events','Named documents made into laws','Any named language','Mentioned dates',]

        for lang, lang_key in zip(languages[:2], language_keys[:2]):
            for ent, ent_key in zip(entities, entity_keys):
                if (lang_key=='ru') &  (ent_key=='PERSON'):
                      ent_key = 'PERS'
                print(f'...processing {lang} entities -- ({ent_key}) -- {ent}...')
                super_dict[embassy_key][f'spacy_ent_{lang_key}_{ent_key}'],super_dict[embassy_key][f'spacy_ent_{lang_key}_{ent_key}_timeseries'] = preprocess_tweets.process_entities(embassy_df, language=lang_key, entity_type=ent_key, n=30)

    super_df = pd.DataFrame.from_dict(super_dict,orient='index').reset_index().rename(columns={'index':'HANDLE'})
    super_df['HANDLE'] = '@' + super_df['HANDLE']
    final_df = base_df.merge(super_df, how='outer',on='HANDLE')

    return final_df.to_csv(filename)




#populate_network_dictionary(base)
#populate_embassy_dictionary(base, 'embassy_db_SAMPLE2.csv')
populate_embassy_dictionary(base, 'embassy_db_FULL.csv')
# populate_embassy_dictionary(base1, 'embassy_db_1.csv')
# populate_embassy_dictionary(base2, 'embassy_db_2.csv')
# populate_embassy_dictionary(base3, 'embassy_db_3.csv')
# populate_embassy_dictionary(base4, 'embassy_db_4.csv')
# populate_embassy_dictionary(base4, 'embassy_db_5.csv')

#print(db_call.get_all_user_activity('StateDept'))
