import pandas as pd
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel,LdaMulticore


def find_topics(tokens, num_topics):
    """
    tokens: an iterable each of whose items is a list of tokens
    num_topics: integer
    
    """
    
    dictionary = Dictionary(tokens) 
    
    corpus = [dictionary.doc2bow(d) for d in tokens]
    # create gensim's LDA model 
    lda_model = LdaModel(corpus, num_topics=num_topics, alpha='auto', chunksize=2000,id2word=dictionary,
                         eval_every=None,passes=20, iterations=400, eta='auto')
    
    
    return lda_model.top_topics(corpus) 

def find_topics_faster(tokens, num_topics):
    """
    tokens: an iterable each of whose items is a list of tokens
    num_topics: integer
    
    """
    
    dictionary = Dictionary(tokens) 
    
    corpus = [dictionary.doc2bow(d) for d in tokens]
    # create gensim's LDA model 
    lda_model = LdaMulticore(corpus, num_topics=num_topics, alpha='auto', chunksize=2000,id2word=dictionary,
                         eval_every=None,passes=20, iterations=400, eta='auto',workers=6)
    
    
    return lda_model.top_topics(corpus) 

    