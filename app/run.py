import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
#import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # [1] show relative counts for each message genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # [2] show relative frequency for each message category
    counts = df.iloc[:, -36:].sum().sort_values().reset_index()
    counts['rf'] = counts[0]/df.shape[0]

    category_counts = pd.Series(data=counts['rf'].values, index=counts['index'])
    category_names = list(category_counts.index)

    # [3] show relative frequency of top 5 words in messages of medical categories
    med_df_results = pd.read_csv('../data/medical_data_top_words.csv', index_col= 0)
    word_counts = pd.Series(data=med_df_results['rf'].values, index=med_df_results.index)
    word_names = list(word_counts.index)


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # Chart [1]
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Chart [2]
        {
            'data': [
                Bar(
                    x=category_counts,
                    y=category_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Relative Frequency of Message Categories',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Relative Frequency"
                }
            }
        },

        # Chart [3]
        {
            'data': [
                Bar(
                    x=word_counts,
                    y=word_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Relative Frequency of Top Words in Medical Category Messages',
                'yaxis': {
                    'title': "Word"
                },
                'xaxis': {
                    'title': "Relative Frequency"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # test: 'People from Jacmel are requesting a tractor in that area because there is a civil unrest and social disturbance in that area.'
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
