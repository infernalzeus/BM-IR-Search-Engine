from src import index_data
from flask import Flask, request, render_template
import random


DATASET_PATH = "data/MINDsmall_train/news.tsv"
INDEX_PATH = "data/index/news.pickle"

df = index_data.create(DATASET_PATH, INDEX_PATH)
model, data_index = index_data.load(INDEX_PATH)


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
             'title': [], 'abstract': [], 'link': [], 'fraud_score': []}

    for i, s in zip(results, scores):
        ans = df.iloc[i]
        final['query_match_score'].append(s)
        final['title'].append(ans["title"])
        final['abstract'].append(ans["abstract"])
        final['link'].append(ans["url"])
        final['fraud_score'].append('empty for now')

    return render_template('search_results.html', final=final, query=query)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5007)
