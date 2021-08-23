import pandas as pd
import numpy as np
from topic_helper_functions import *

# requires lots of memory (~16 gb), will add db call functions later to reduce the memory cost
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

rus_emb_ids.extend(['all'])
usa_emb_ids.extend(['all'])

all_rus_data = data[data['user_id'].isin(rus_emb_ids)]
all_USA_data = data[data['user_id'].isin(usa_emb_ids)]

all_rus_data = all_rus_data.dropna(subset=['clean_text'])
all_USA_data = all_USA_data.dropna(subset=['clean_text'])

all_rus_data['created_at'] = pd.to_datetime(all_rus_data['created_at'])
all_rus_data['year'] = all_rus_data['created_at'].dt.year

all_USA_data['created_at'] = pd.to_datetime(all_USA_data['created_at'])
all_USA_data['year'] = all_USA_data['created_at'].dt.year

all_rus_data = all_rus_data.astype({'lemmas':'str'})
all_rus_data['lemmas'] = all_rus_data['lemmas'].apply(lister)

all_USA_data = all_USA_data.astype({'lemmas':'str'})
all_USA_data['lemmas'] = all_USA_data['lemmas'].apply(lister)


all_rus_years = list(all_rus_data['year'].unique())
all_USA_years = list(all_USA_data['year'].unique())
all_rus_years.extend(['all'])
all_USA_years.extend(['all'])

all_rus_lang = list(all_rus_data['lang'].unique())
all_USA_lang = list(all_USA_data['lang'].unique())
all_rus_lang.extend(['all'])
all_USA_lang.extend(['all'])

# uncomment following if using jupyter environment
# !echo 'source','ids','num_topics','languages','years' > assets/RUS_topic_log.csv
# !echo 'source','ids','num_topics','languages','years' > assets/USA_topic_log.csv

for num_topics in range(2,10):
     generate_topic_data(all_rus_data,'RUS',rus_emb_ids,num_topics,all_rus_lang,all_rus_years)
    
for num_topics in range(7,10):
    generate_topic_data(all_USA_data,'USA',usa_emb_ids,num_topics,all_USA_lang,all_USA_years)
    

