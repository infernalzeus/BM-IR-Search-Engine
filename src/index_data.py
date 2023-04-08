import string
import pickle
from nltk.corpus import stopwords
import pandas as pd
from src import index_data
from src.model import BM25Okapi
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')


stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()  # convert to lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stop words
    return text

def load(idxpath):
    with open(idxpath, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data['model'], pkl_data['data_index']


def create(fpath, idxpath):
    dataset_file = fpath
    COL_NAMES = ["doc_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    df = pd.read_csv(dataset_file, sep="\t", header=None, names=COL_NAMES)
    print(f"Data dimensions: {df.shape}")

    # drop when df["title"] & df["abstract"] are nan
    df_no_nan = df.dropna(subset=["title", "abstract"])
    print(f"Data dimensions after removing nan: {df_no_nan.shape}")

    # preprocess the text
    df_no_nan["text"] = df_no_nan["title"] + " " + df_no_nan["abstract"]
    df_no_nan["preprocessed_text"] = df_no_nan["text"].apply(index_data.preprocess_text)
    corpus = df_no_nan["preprocessed_text"].tolist()

    # Create and save index
    model = BM25Okapi(corpus)
    pkl_data = {
        "model": model,
        "data_index": df_no_nan.index,
    }
    with open(idxpath, "wb") as f:
        pickle.dump(pkl_data, f)

    return df