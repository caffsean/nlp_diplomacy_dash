import pandas as pd
import numpy as np
from topic_model import find_topics
import re
import pickle as pkl
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import LdaModel,LdaMulticore

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")




def find_topics_faster(tokens, num_topics):
    """
    tokens: an iterable each of whose items is a list of tokens
    num_topics: integer
    Output - topics and words associated with them
    """
    
    dictionary = Dictionary(tokens) 
    
    corpus = [dictionary.doc2bow(d) for d in tokens]
    # create gensim's LDA model 
    lda_model = LdaMulticore(corpus, num_topics=num_topics, chunksize=2000,id2word=dictionary,
                         eval_every=None,passes=20, iterations=400, eta='auto',workers=6)
    
    
    return lda_model.top_topics(corpus)

def lister(string):
    '''
    It does some final clean up to a string
    removes numbers, whitespaces, newline and returns a list of words
    
    '''
    string = re.sub('[0-9]+', '', string)
    string = re.sub(r'[|&$!@#%*]', '', string)
    string = string.strip()
    string = string.replace('\n','')
    string = string.replace('nan','')
    final = string.split(',')
    final = [str.strip(i) for i in final]
    final = [i for i in final if i!='/']
    final = [i for i in final if i!='']
    final = [i for i in final if i!=' ']
    final = [i for i in final if len(i)>1]
    return final

def data_frame_maker(topics_dict, num_words):
    '''
    Takes and topic dictionary and returns an array of labels and dataframe with scores for heatmap
    
    '''
    words = []
    word_scores = []
    topics = []
    topic_scores = []
    for idx in range(len(topics_dict)):
        column_for_word_level = []
        column_for_word_names = []
        for idx2 in range(len(topics_dict[idx][0][:num_words])):
            column_for_word_level.append(np.round(topics_dict[idx][0][idx2][0],4))
            column_for_word_names.append(topics_dict[idx][0][idx2][1])# + "\n" + str(np.round(topics_dict[idx][0][idx2][0],4)))
        topics.append(f'Topic {idx+1}\n{np.round(topics_dict[idx][1],4)}')
        word_scores.append(column_for_word_level)
        words.append(column_for_word_names)
    labels = np.array(words)   
    df = pd.DataFrame(word_scores, index=topics)
    return labels, df 

# quick helper functions
def get_profile_name(ids):
    data = profiles[profiles['user_id']==ids]
    print(str(data['screen_name'].iloc[0]))
    
def year_data_filter(df,year):
    if year == 'all':
        return df
    else:
        final = df[df['year']==year]
        return final

def lang_data_filter(df,lang):
    if lang == 'all':
        return df
    else:
        final = df[df['lang']==lang]
        return final
    
def emb_data_filter(df,emb_id):
    if emb_id == 'all':
        return df
    else:
        final = df[df['user_id']==emb_id]
        return final
    
def generate_topic_data(tweets_df,source,target,num_topics,lang,year):
    final_dict = {}
    counter = 0
    second_counter =1
    for ids in target:
        temp_df = emb_data_filter(tweets_df,ids)

        for languages in lang:
            lang_temp_df = lang_data_filter(temp_df,languages)
            if lang_temp_df.empty:
                continue
            for years in year:
                year_temp_df = year_data_filter(lang_temp_df,years)
                if year_temp_df.empty:
                    continue
#                 print(f"working on {source}_{ids}_{num_topics}_topics_{languages}_lang_{years}_year data")
                !echo f"working on {source}_{ids}_{num_topics}_topics_{languages}_lang_{years}_year data" >>log.txt
                try:
                    topic_data = find_topics(year_temp_df['lemmas'],num_topics)
                except ValueError:
                    continue
                counter+=1
                if counter==100:
                    second_counter = 1
                    final_counter = counter*second_counter
                    counter = 0
                    print(f'completed {final_counter} files...')
                    
                label,final_df = data_frame_maker(topic_data,15)

                final_dict['label'] = label
                final_dict['df'] = final_df
                final_dict['og_data'] = topic_data
                !echo {source},{ids},{num_topics},{languages},{years} >> assets/{source}_topic_log.csv
                filename = f"assets/topic_data/{source}_{ids}_{num_topics}_topics_{languages}_lang_{years}_year.pkl"

                file_to_write = open(filename, "wb")

                pkl.dump(final_dict,file_to_write)
    
    








