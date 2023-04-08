from src import index_data


if __name__ == "__main__":
    DATASET_PATH = "data/MINDsmall_train/news.tsv"
    INDEX_PATH = "data/index/news.pickle"

    df = index_data.create(DATASET_PATH, INDEX_PATH)
    model, data_index = index_data.load(INDEX_PATH)
    while True:
        query = input("Enter your query: ")
        # preprocess and query
        tokenized_query = index_data.preprocess_text(query).split()

        # process
        results, scores = model.get_top_n(tokenized_query, data_index, n=10)
        for i, s in zip(results, scores):
            ans = df.iloc[i]
            print("Query match score:", s)
            print("Title:", ans["title"])
            print("\tAbstract:", ans["abstract"])
            print("\tLink to full article:", ans["url"])
            print()
