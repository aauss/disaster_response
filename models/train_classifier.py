import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import nltk
import pandas as pd
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Loads data from sqlite data base and splits into predictors and targets. 

    Args:
        database_filepath: file path to sqllite database

    Returns:
        Predictors, targets and target names
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql(f"SELECT * FROM {Path(database_filepath).stem}", engine)
    X = df.loc[:, ["message", "original", "genre"]]
    Y = df.loc[:, ~df.columns.isin(["message", "original", "id", "genre"])]
    category_names = Y.columns
    return X, Y, category_names


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


def build_model() -> GridSearchCV:
    """Builds text classification model.

    Text classifier uses a tfidf-vectorization with a custom tokenizer
    to train a multinomial naive Bayes classifier.

    Returns:
        Text classifier
    """
    pipeline = Pipeline(
        [
            ("count_vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(MultinomialNB())),
        ]
    )

    parameters = {
        "count_vect__ngram_range": ((1, 1), (1, 2)),
        "tfidf__use_idf": (True, False),
        "clf__estimator__alpha": [0.1, 0.5, 1],
    }

    return GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)


def evaluate_model(
    model: GridSearchCV,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    category_names: List[str],
) -> None:
    """Evaluates a model per predicted class printing several metrics.

    Args:
        model: A model pipeline of a text classifier.
        X_test: A dataframe containing the model predictors
        Y_test: A dataframe containing the model targets
        category_names: List of targets
    """
    y_pred = model.predict(X_test["message"])
    categories = []
    f1s = []
    precisions = []
    recalls = []
    for i, category in enumerate((category_names)):
        categories.append(category)
        f1s.append(f1_score(Y_test[category], y_pred[:, i]))
        precisions.append(precision_score(Y_test[category], y_pred[:, i]))
        recalls.append(recall_score(Y_test[category], y_pred[:, i]))
    results = pd.DataFrame(
        {
            "category": categories,
            "F1": f1s,
            "Precision": precisions,
            "Recalls": recalls,
        }
    )
    results.to_csv(Path(__file__).resolve().parent / "results.csv", index=False)
    print(results, end="\n\n")
    print("Results saved under results.csv")


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """Save a machine learning model.

    Args:
        model: 
        model_filepath: [description]
    """
    with open(f"{model_filepath}", "wb") as f:
        pickle.dump(model, f)


def main() -> None:
    """ Trains classifier using processed messages from database.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train["message"], Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
