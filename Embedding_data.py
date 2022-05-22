# Импорт необходимых библиотек 

import numpy as np 
import pandas as pd 
import seaborn 

from natasha import MorphVocab, Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, Doc 

# Download the data from the file `train.tsv`

FILE_PATH = 'D:/Обучение/ODS MLOps/Project_MLOps/Project_MLOps/Data/train.tsv' 

def docVocab(text): 
    """ 
    Function for getting a list of tokens for the Russian language.
    Note that our input data is in Russian and 
    the words have different endings.
    We will reduce them to one form using lemmatization, 
    and also use the results of syntactic analysis,
    let's take into account the connections between words, 
    normalize named entities. This will help us Natasha.
    """ 
    emb = NewsEmbedding() 
    morph_vocab = MorphVocab() 
    segmenter = Segmenter() 
    morph_tagger = NewsMorphTagger(emb) 
    syntax_parser = NewsSyntaxParser(emb) 
    ner_tagger = NewsNERTagger(emb) 

    doc = Doc(text) 
    doc.segment(segmenter) 
    doc.tag_morph(morph_tagger) 
    
    for token in doc.tokens:
        token.lemmatize(morph_vocab) 
        
    lemms = {_.text: _.lemma for _ in doc.tokens if _.pos != 'PUNCT'} 
    
    doc.parse_syntax(syntax_parser) 
    doc.tag_ner(ner_tagger) 
    
    for span in doc.spans:
        span.normalize(morph_vocab)
    
    spans = {_.text: _.normal for _ in doc.spans} 
    lemms.update(spans) 
    
    return list(lemms.values()) 


def news_embedding(data) -> pd.DataFrame: 
    """
    Function removes excess columns and enforces 
    correct data types. 
    :param df: Original DataFrame 
    :return: Updated DataFrame 
    """
    X = data.title 
    y = data.is_fake 

    X = X.apply(lambda x: ' '.join(docVocab(x))) 

    data.title = X 
    data.is_fake = y 

    data.is_fake = y 

    return data 

if __name__ == '__main__': 
    data = pd.read_csv(FILE_PATH, sep="\t") 
    data = news_embedding(data) 
    data.to_csv('data.tsv', sep='\t', index=False) 