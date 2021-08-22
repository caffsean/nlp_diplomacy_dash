from typing import final
import pandas as pd
import numpy as np
import os
import pickle as pkl
import networkx as nx

def generate_mention_df():
    '''
    Generates mention df from comention data
    Output: final_df - comention dataframe
    '''
    try:
        final_df = pkl.load(open('assets/network_data/mention_df_all.pkl','rb'))
    except:
        mentions = pd.read_csv('tweet_mentions.csv')
        id_screenname_dict = pkl.load(open('assets/id_screenname_dict.pkl', "rb"))
        tweet_fav_count = pkl.load(open('assets/tweet_fav_count.pkl', "rb"))
        tweet_ret_count = pkl.load(open('assets/tweet_ret_count.pkl', "rb"))

        mentions['screen_name']=mentions['user_id'].map(id_screenname_dict)
        mentions['mention_screen_name'] = mentions['mentioned_user_id'].map(id_screenname_dict)
        mentions['fav_count'] = mentions['tweet_id'].map(tweet_fav_count)
        mentions['ret_count'] = mentions['tweet_id'].map(tweet_ret_count)

        mentions = mentions.dropna(subset=['mention_screen_name'])
        mentions = mentions.fillna(0)

        final_df = mentions.groupby(['tweet_id']).agg({'screen_name':lambda x: [name for name in x.unique()],
                                                        'mention_screen_name':lambda x: [name for name in x.unique()],
                                                        'fav_count': np.average,
                                                        'ret_count':np.average})

        final_df['num']= final_df['mention_screen_name'].str.len()

        final_df['all']=final_df['screen_name']+final_df['mention_screen_name']
        pkl.dump(final_df,open('assets/network_data/mention_df_all.pkl', "wb"))
    return final_df

def mention_network(mention_data):
    '''
    Generates a large network graph based on mention network
    Input: mention_data - Dataframe containing list of co-mentions
    Output: G_w - returns weighted graph
    '''
    try:
        G_w = pkl.load(open('assets/network_data/mention_graph.pkl','rb'))
    except:
        edges = {}
        for row in mention_data.itertuples():
            screen_names = row[-1]
            for i in range(len(screen_names)):
                for j in range(i,len(screen_names)):
                    if screen_names[i]==screen_names[j]:
                        continue
                    elif screen_names[i]+'-'+screen_names[j] in edges:
                        edges[screen_names[i]+'-'+screen_names[j]]['count'] = edges[screen_names[i]+'-'+screen_names[j]]['count'] + 1
                    elif screen_names[j]+'-'+screen_names[i] in edges:
                        edges[screen_names[j]+'-'+screen_names[i]]['count'] = edges[screen_names[j]+'-'+screen_names[i]]['count'] + 1
                    else:
                        edges[screen_names[i]+'-'+screen_names[j]] = {'count': 1, 'node_1':screen_names[i], 'node_2':screen_names[j]}

        edge_df = pd.DataFrame(edges).T



        edge_list = list(edge_df[['node_1','node_2','count']].itertuples(index=False, name=None))

        G_w = nx.Graph()
        G_w.add_weighted_edges_from(edge_list)
        pkl.dump(G_w,open('assets/network_data/mention_graph_pos.pkl','wb'))
    return G_w

final_df = generate_mention_df()
G = mention_network(final_df)
pos = nx.kamada_kawai_layout(G) # Beware of this as it takes a extremely long time and huge memory requirements >32 GB





