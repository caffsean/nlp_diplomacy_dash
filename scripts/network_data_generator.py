import pandas as pd
import numpy as np
import pickle as pkl
import re
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import sqlalchemy as sa
from sqlalchemy.orm import close_all_sessions
database = 'postgres'
user = 'postgres'
password = 'XXXXXXXXXXXXXX'
host= 'XXXXXXXXXXXXX'

engine = sa.create_engine(f'postgresql://{user}:{password}@{host}/{user}')

def extract_interesting_ent(row):
    '''
    extract intereting entities from the list
    input-  row : dataframe row
    output- final : output list of entities
    '''

    interest = ['PERSON','GPE','ORG','NORP']
    final_idx = []
    final_ent = []
    entities = row['entities']
    entities_labels = row['entities_labels']
    
    for i in range(len(entities_labels)):
        if entities_labels[i] in interest:
            final_idx.append(i)
    
    if final_idx:
        for idx in final_idx:
            try:
                # this loop is in case there are any mismatches in indexes
                final_ent.append(entities[idx])
            except IndexError:
                continue
    else:
        return None
    return final_ent

def lister(string):
    '''
    It does some final clean up to a string
    removes numbers, whitespaces, newline and returns a list of words
    
    input-  string : string from database lemmas table
    output- final : output list of entities
    '''
    
    string = re.sub('[0-9]+', '', string)
    string = string.strip()
    string = string.replace('\n','')
    string = string.replace('nan','')
    final = string.split(',')
    final = [i for i in final if i]
    final = [i for i in final if i!='/']
    final = [i for i in final if i!=' ']
    return final

def get_user_data(user_id):
    '''
    Given user id it generated nodes from entities mentioned in their tweets, 
    along with edge information
    
    input-  user_id : interested user_id
    output- all_nodes : list of nodes
            edge_df : dataframe containing source, target and weight
            edge_list : list of edges in tuple form (u,v,w)
    '''
    # uncomment the following if using databse connection and local memory is limited
    
    #     tweets_df = pd.read_sql_query(f"SELECT *  \
    #     FROM tweets\
    #     WHERE (user_id = ({user_id}) \
    #     AND (tweets.entities IS NOT NULL)) " , con=engine)
    
    # load data locally to have faster performance. Warning: it uses lots of memory (recommended memory 16gb or higher)
    
    tweets_df = data[data['user_id']==user_id]
    tweets_df = tweets_df.dropna(subset=['entities','entities_labels'])
    
    df_entities = tweets_df[['entities', 'entities_labels', 'tweet_id']]
    df_entities.dropna(subset= ['entities'], inplace = True)
    df_entities['entities'] = df_entities['entities'].apply(lister)
    df_entities['entities_labels'] = df_entities['entities_labels'].apply(lister)
    df_entities['entities'] = df_entities.apply(extract_interesting_ent,axis=1)
    df_entities.dropna(subset=['entities'],inplace=True)
    
    res = [(x, i[j + 1]) for i in df_entities.entities for j, x in enumerate(i) if j < len(i) - 1]
    res_sorted = [tuple(sorted(x)) for x in res]
    edge_df = pd.DataFrame(pd.DataFrame({'edges':res_sorted}).value_counts('edges')).reset_index()
    edge_df.rename(columns = {0:'weight'}, inplace = True)
    
    edge_df['final'] = list(zip(edge_df.edges, edge_df.weight))
    #ignore the order of operation here
    
    edge_list = list(edge_df['final'])
    edge_list =[(a, b, c) for (a, b), c in edge_list]
    edge_list = [(a,b,c) for a,b,c in edge_list if a!=' ']
    
    all_nodes = df_entities.explode('entities').entities.unique()
    all_nodes = [i for i in all_nodes if i!=' ']
    edge_df['target'] = edge_df['edges'].apply(lambda x : x [0])
    edge_df['source'] = edge_df['edges'].apply(lambda x : x [1])
    edge_df = edge_df[edge_df['target']!='']
    edge_df = edge_df[edge_df['source']!='']
    edge_df = edge_df[edge_df['target']!=' ']
    edge_df = edge_df[edge_df['source']!=' ']
    return all_nodes, edge_df, edge_list

def random_char(y):
    import random
    import string
    return ''.join(random.choice(string.ascii_letters) for x in range(y))
    
def build_network(nodes_list, edges_df, edge_list,centrality_measure):
    '''
    Generates network element data for dash cytoscape
    
    input-  nodes_list: list of nodes
            edge_df : dataframe containing source, target and weight
            edge_list : list of edges in tuple form (u,v,w)
            centrality_measure : centrality measure to use for node sizes
    output- network_elements : elemnt file for dash
            G : network graph
            pos : position data (default : Kamada Kawai layout)
    
    '''
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_weighted_edges_from(edge_list)

    pos = nx.kamada_kawai_layout(G)

    df_nodes  = pd.DataFrame.from_dict(pos).T.reset_index()
    
    df_nodes.columns = ['node', 'x', 'y']
    network_node_imp_measure = centrality_measure(G)
    df_nodes['deg_centrality'] = df_nodes['node'].map(network_node_imp_measure)
    df_nodes['x'] = df_nodes['x']*2000 # multiplier for visual esthetic
    df_nodes['y'] = df_nodes['y']*2000

    df_nodes.dropna(subset=['node'],inplace=True)
    nodes =[]
    for index, row in df_nodes.iterrows():
        size = int(np.round(row['deg_centrality']*500, 0))
        nodes.append({'data':{'id' : row['node'], 'label':row['node'], 'size': size }, 
                       'position':{'x':row['x'], 'y':row['y']}
                      })
    edge_class = random_char(6)
    edges = []
    for index, row in edges_df.iterrows():
        source, target,weight  = row['source'],row['target'],row['weight']
        edges.append({'data':{'source': source, 'target':target, 'weight': weight}, 'classes':edge_class, 'size':weight})
    network_elements = nodes
    network_elements.extend(edges)
    
    return network_elements,G,pos

# shell commands are for jupyter environment, use subprocess for logging

def simulate_SI(G, importance_measure=None, iterate=100, n=1, beta=0.1):
    '''
    Simulates SI model
    
    input-  G: network graph
            importance_measure : Include centrality measure
            iterate : how many times to simulate
            n : number of intial infected
            beta : infection probability
    output- list of how many infected in each iterations
    
    '''
    if importance_measure:
        # select seed nodes
        sorted_node = sorted(importance_measure(G).items(), key=operator.itemgetter(1))[::-1]
        highest_node = [n for n, _ in sorted_node[:n]]

    # Model selection
    model = ep.SIModel(G)
    cfg = mc.Configuration()
    frac =float(n)/len(G.nodes)
    cfg.add_model_parameter('beta',beta)
    
    if importance_measure:
        cfg.add_model_initial_configuration("Infected", highest_node)
        
    else:
        cfg.add_model_parameter("fraction_infected",frac)

    model.set_initial_status(cfg)
    
    # Simulation execution
    iterations = model.iteration_bunch(iterate)
    return [it['node_count'][1] for it in iterations]

def generate_network_data(source,target,measure,SI_diffusion_model=False):
    '''
    Generates network element data for dash cytoscape
    
    input-  Source : source country
            target : embassy/entity id for the country
            measure : centrality measure to use for node sizes
            SI_diffusion_model : generates the diffusion model for the network [default : False]
    output- saves network element, graph, position and diffusion model results in a pickle file
    
    '''
    if measure=='betweenness':
        centrality_measure = nx.betweenness_centrality
    elif measure=='closeness':
        centrality_measure = nx.closeness_centrality
    elif measure=='degree':
        centrality_measure= nx.degree_centrality
    elif measure=='eigenvector':
        centrality_measure = nx.eigenvector_centrality
    elif measure=='current':
        centrality_measure = nx.current_flow_betweenness_centrality
    elif measure=='pagerank':
        centrality_measure = nx.pagerank
    else:
        print('pick from closeness, degree, eigenvector, and current flow betweenness centrality measures!')
        # Use subprocess or uncomment if using jupyter notebook
        #!echo f"{d1}- Error picking the centrality measure, measure requested '{measure}''" >> generate_network_data_error_log.txt
    print(f"Working on {source} country's embassy with ID {target} and node sizes with {measure} centrality measure...")
    nodes_list, edge_df, edge_list = get_user_data(target)
    network_elements,G,pos = build_network(nodes_list,edge_df,edge_list,centrality_measure)
    
    transitivity_score = nx.transitivity(G)
    avg_clustering_coef = nx.average_clustering(G)
    hub, authority = nx.hits(G, max_iter=200)
    hub = sorted(hub.items(), key=lambda item: item[1],reverse = True)
    authority = sorted(authority.items(), key=lambda item: item[1],reverse = True)
    h5 = [x[0] for x in hub[1:6]]         # a list of the 5 nodes with the largest hub scores
    a5 = [x[0] for x in authority[1:6]]   # a list of the 5 nodes with the largest authority scores
    network_measure_dict = {}
    network_measure_dict['transitivity']=transitivity_score
    network_measure_dict['avg_clustering_coefficient']=avg_clustering_coef
    network_measure_dict['hub5']=h5
    network_measure_dict['auth5'] = a5
    network_measure_dict['hub']=hub
    network_measure_dict['auth'] = authority
    
    pkl.dump(network_measure_dict,open(f"assets/network_data/measures/{source}_{target}_network_{measure}.pkl",'wb'))
    
    if SI_diffusion_model:
        diff_model = simulate_SI(G,measure,400,0.1)
        pkl.dump(diff_model,open(f"assets/network_data/diff_model/{source}_{target}_SI_model.pkl",'wb'))
    # Use subprocess or uncomment if using jupyter notebook
    #!echo {source},{target},{measure} >> assets/{source}_{measure}_network_log_final.csv
    
    filename1 = f"assets/network_data/{measure}/{source}_{target}_network_{measure}_new.pkl"
    file_to_write1 = open(filename1, "wb")
    pkl.dump(network_elements,file_to_write1)
    
    filename2 = f"assets/network_data/graphs/{source}_{target}_graph.pkl"
    file_to_write2 = open(filename2, "wb")
    pkl.dump(G,file_to_write2)
    
    filename3 = f"assets/network_data/pos/{source}_{target}_pos.pkl"
    file_to_write3 = open(filename3, "wb")
    pkl.dump(pos,file_to_write3)


# if using db calls instead comment the following line
data = pd.read_csv('tweets.csv', low_memory=False)

profiles = pd.read_csv('profiles.csv',usecols=['user_id','screen_name'],low_memory=False)
embassies = pd.read_csv('base_info_capitals.txt').rename({'HANDLE':'screen_name'},axis=1)
embassies['screen_name'] = embassies['screen_name'].str.replace('@','')
embassies = embassies.dropna(subset=['screen_name'])
embassies = embassies.merge(profiles,how='left',on='screen_name')
embassies = embassies.dropna(subset=['user_id'])
embassies = embassies.astype({'user_id':'int64'})
all_rus = embassies[embassies['Source']=='RUS']
all_USA = embassies[embassies['Source']=='USA']
rus_emb_ids = list(all_rus['user_id'].unique())
usa_emb_ids = list(all_USA['user_id'].unique())

rus_emb_ids.extend([343933160]) # including mfa_russia
usa_emb_ids.extend([9624742]) # including US state_dept



import os.path
network_measure = input('Please enter a centrality measure: ')
############# USA network generation #############
for ids in usa_emb_ids:
    if os.path.isfile(f"assets/network_data/{network_measure}/USA_{ids}_network_{network_measure}_new.pkl"):
        print(f"graph data for USA -> {ids} exist! Skipping...")
        # Use subprocess or uncomment if using jupyter notebook
        #!echo f"graph data for USA -> {ids} exist! Skipping..." >> usa_network_log_overnight_8_20_21.txt
        continue
    else:
        try:
            generate_network_data('USA',ids,network_measure)
        except ValueError:
            ''' ValueErrors generates when the graph is too small'''
            # Use subprocess or uncomment if using jupyter notebook
            #!echo f"USA -> {ids} isn't completed for too small graph" >> usa_network_log_overnight_8_20_21.txt
            continue
        except:
            '''Another potential error generates when hits alg fails to converge'''
            # Use subprocess or uncomment if using jupyter notebook
            #!echo f"USA -> {ids} isn't completed for failed convergence" >> usa_network_log_overnight_8_20_21.txt
            continue
            
            
############# RUS network generation #############

for ids in rus_emb_ids:
    if os.path.isfile(f"assets/network_data/{network_measure}/RUS_{ids}_network_{network_measure}_new.pkl"):
        print(f"graph data for RUS -> {ids} exist! Skipping...")
        # Use subprocess or uncomment if using jupyter notebook
        #!echo f"graph data for RUS -> {ids} exist! Skipping..." >> rus_network_log_overnight_8_20_21.txt
        continue
    else:
        try:
            generate_network_data('RUS',ids,network_measure)
        except ValueError:
            # Use subprocess or uncomment if using jupyter notebook
            #!echo f"RUS -> {ids} isn't completed" >> rus_network_log_overnight_8_20_21.txt
            continue
        except:
            # Use subprocess or uncomment if using jupyter notebook
            #!echo f"RUS -> {ids} isn't completed for failed convergence" >> rus_network_log_overnight_8_20_21.txt
            continue






