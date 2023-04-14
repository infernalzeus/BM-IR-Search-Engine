from src import index_data
from src.model import get_sentiment
from flask import Flask, request, render_template
import flask.cli
import random


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('query')

    processed_query = index_data.preprocess_text(query)

    # process
    results, scores = model.get_top_n(
        processed_query, data_index, n=random.randint(15, 50))

    final = {'query_match_score': [],
             'title': [], 'abstract': [], 'link': [],
             'sentiment_score': [], 'subjectivity_score': [],
             'news_category': []}

    for i, s in zip(results, scores):
        ans = df.iloc[i]
        if s != 0:
            final['query_match_score'].append(s)
            final['title'].append(ans["title"])
            final['abstract'].append(ans["abstract"])
            final['link'].append(ans["url"])
            sentiment_score, subjectivity_score = get_sentiment(
                ans["title"] + ' ' + ans['abstract'])
            final['sentiment_score'].append(sentiment_score)
            final['subjectivity_score'].append(subjectivity_score)
            final['news_category'].append(
                ans['category'].title() or 'No category')

    return render_template('search_results.html', final=final, query=query)


if __name__ == "__main__":
    DATASET_PATH = "data/MINDlarge_train/news.tsv"
    INDEX_PATH = "data/index/news.pickle"

    df = index_data.create(DATASET_PATH, INDEX_PATH)
    model, data_index = index_data.load(INDEX_PATH)

    flask.cli.show_server_banner = lambda *args: None

    app.run(debug=False, host='0.0.0.0', port=5005)
