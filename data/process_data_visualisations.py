import re

import sys
import pandas as pd
from sqlalchemy import create_engine

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist


def frequency_distribution(text, n):
    '''
    Corpus tokenization, lemmatising, calculation of frequency distributions and returning the n most common.
    '''
    # Normalize text
    normalized = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Word tokenize
    words = word_tokenize(normalized)
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # Calculate frequency distribution
    freq = FreqDist(lemmed)

    return freq.most_common(n)


def medical_graph(df):
    med_df = df[(df['medical_help']==1) | (df['medical_products']==1) | (df['hospitals'] == 1)]
    med_corpus = " ".join(med_df.message.values)
    med_dist = frequency_distribution(med_corpus, 5)
    top_words = [w[0] for w in med_dist]
    top_freq = [w[1] for w in med_dist]

    med_df_results = pd.DataFrame(data=top_freq, index=top_words)
    med_df_results['rf'] = med_df_results[0]/med_df.shape[0]

    return med_df_results

def save_data(df, file):
    df.to_csv(file)


def main():

    print('Loading data...\n')
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('Messages', engine)

    print('Generating data for visualisation...')
    df = medical_graph(df)

    filename = 'medical_data_top_words.csv'
    print('Saving data...\n    DATA: {}'.format(filename))
    save_data(df, filename)

    print('Graph data saved to csv!')



if __name__ == '__main__':
    main()
