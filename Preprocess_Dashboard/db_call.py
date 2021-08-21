'''
This file contains database calls which are used to retrieve that data to be preprocessed
'''
import pandas as pd
import numpy as np

database = 'postgres'
user = 'postgres'
password = 'rKKFiDXpiu6Wbv3'
host='47.200.121.209'


## Get all tweets from user
def table_query_get_tweets(user_name):
    from sqlalchemy import create_engine
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{user}')
    #table_data = pd.read_sql_query(f"SELECT * FROM twitter_profile_stats WHERE user_id in (SELECT user_id FROM twitter_profiles WHERE screen_name in {emb})",con=engine)
    table_data = pd.read_sql_query(f"SELECT * FROM tweets WHERE user_id in (SELECT user_id FROM twitter_profiles WHERE screen_name = ('{user_name}'))",con=engine)
    return table_data

## Get all info from user
def get_all_user_info(user_name):
    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{user}')
    user_info =  pd.read_sql_query(f"SELECT * FROM twitter_profiles WHERE screen_name = ('{user_name}')",con=engine)#['user_id'][0]
    return user_info

## Get all user stats
def get_all_user_stats(user_name):
    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{user}')
    user_id =  pd.read_sql_query(f"SELECT user_id FROM twitter_profiles WHERE screen_name = ('{user_name}')",con=engine)['user_id'][0]
    user_info =  pd.read_sql_query(f"SELECT * FROM twitter_profile_stats WHERE user_id = ('{user_id}')",con=engine)#['user_id'][0]
    return user_info

## Get user ID
def get_user_id(user_name):
    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{user}')
    user_id =  pd.read_sql_query(f"SELECT user_id FROM twitter_profiles WHERE screen_name = ('{user_name}')",con=engine)['user_id'][0]
    return user_id
## Get MULTIPLE user IDs
def get_user_id_multiple(user_names):
    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{user}')
    user_names = tuple(user_names)
    user_id =  pd.read_sql_query(f"SELECT user_id FROM twitter_profiles WHERE screen_name in {user_names}",con=engine)['user_id'][0]
    return user_id
## Get all user activity
def get_all_user_activity(user_name):
    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{user}')
    user_id =  pd.read_sql_query(f"SELECT user_id FROM twitter_profiles WHERE screen_name = ('{user_name}')",con=engine)['user_id'][0]
    sql_query = f"SELECT * FROM tweets WHERE (tweet_id IN (SELECT tweet_id FROM retweets WHERE user_id = ('{user_id}'))) OR (user_id = ('{user_id}'))"
    #table_data = pd.read_sql_query(sql_query, con=engine, chunksize=4000)
    dfs = []
    for chunk in pd.read_sql_query(sql_query, con=engine, chunksize=1000):
	        dfs.append(chunk)
    table_data = pd.concat(dfs)
    return table_data

## Get all hashtags from user
def get_hashtags_from_user(user_name):
    from sqlalchemy import create_engine
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{user}')
    user_id =  pd.read_sql_query(f"SELECT user_id FROM twitter_profiles WHERE screen_name = ('{user_name}')",con=engine)['user_id'][0]
    tweet_hashtags = pd.read_sql_query(f"SELECT created_at, tweets.tweet_id, hashtags.hashtag \
                                    FROM tweets \
                                    right JOIN tweet_hashtags \
                                    ON tweets.tweet_id =tweet_hashtags.tweet_id \
                                    left JOIN hashtags \
                                    ON tweet_hashtags.hashtag_id=hashtags.hashtag_id \
                                    WHERE tweets.user_id = ('{user_id}')",con=engine)
    return tweet_hashtags
## Get all user mentions from user
def get_mentions_from_user(user_name):
    from sqlalchemy import create_engine
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{user}')
    mentions_query = f"SELECT twitter_profiles.screen_name, tweets.tweet_id, tweets.created_at \
                       FROM tweet_mentions \
                       LEFT JOIN tweets \
                       ON tweet_mentions.tweet_id = tweets.tweet_id\
                       LEFT JOIN twitter_profiles \
                       ON tweet_mentions.mentioned_user_id = twitter_profiles.user_id \
                       WHERE tweet_mentions.user_id = \
                       (SELECT twitter_profiles.user_id FROM twitter_profiles WHERE screen_name = ('{user_name}'))"
    table_data = pd.read_sql_query(mentions_query, con=engine)
    return table_data
