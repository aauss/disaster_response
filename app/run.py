import json
import pickle
from typing import List

import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text: str) -> List[str]:
    """Tokenized a text.

    Args:
        text: Text to be tokenized.

    Returns:
        List of tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(token).lower().strip() for token in tokens]


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("DisasterResponse", engine)

# load model
with open("../models/classifier.pkl", "rb") as f:
    model = pickle.load(f)


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    target_counts = (
        df.loc[:, ~df.columns.isin(["id", "message", "original", "genre"])]
        .sum()
        .sort_values(ascending=False)
    )
    target_names = list(target_counts.index)

    text_len_bins = {
        "0-100": df["message"].str.len().between(left=0, right=10).sum(),
        "101-500": df["message"].str.len().between(left=11, right=100).sum(),
        "501-1000": df["message"].str.len().between(left=101, right=1000).sum(),
        "1001-": df["message"].str.len().between(left=1001, right=9999999).sum(),
    }

    # create visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [Bar(x=target_names, y=target_counts)],
            "layout": {
                "title": "Distribution of Targets",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Target"},
            },
        },
        {
            "data": [Bar(x=list(text_len_bins.keys()), y=list(text_len_bins.values()))],
            "layout": {
                "title": "Message Lengths",
                "yaxis": {"title": "Count", "type": "log"},
                "xaxis": {"title": "Message Length"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
