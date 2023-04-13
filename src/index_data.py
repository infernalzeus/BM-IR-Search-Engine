"""
Creates the Index and prepares index for the BM25 models
"""
import string
import pickle
from nltk.corpus import stopwords
import pandas as pd
import nltk
from src.model import BM25Okapi
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


stop_words = set(stopwords.words("english"))

pd.options.mode.chained_assignment = None

def preprocess_text(text):
    """Preprocess text, used before indexing and query search
    """
    text = text.lower()  # convert to lowercase
    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # remove stop words
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def load(idxpath):
    """Loads the pickle file.
    """
    with open(idxpath, "rb") as idx_file:
        pkl_data = pickle.load(idx_file)
    return pkl_data['model'], pkl_data['data_index']


def create(fpath, idxpath):
    """ Creates index and returns the corpus dataframe
    """
    dataset_file = fpath
    col_names = ["doc_id", "category", "subcategory", "title",
                 "abstract", "url", "title_entities", "abstract_entities"]
    df_original = pd.read_csv(dataset_file, sep="\t",
                              header=None, names=col_names)
    print(f"Data dimensions: {df_original.shape}")

    # drop when df["title"] & df["abstract"] are nan
    df_no_nan = df_original.dropna(subset=["title", "abstract"])
    print(f"Data dimensions after removing nan: {df_no_nan.shape}")

    # preprocess the text
    df_no_nan["text"] = df_no_nan["title"] + " " + df_no_nan["abstract"]
    df_no_nan["preprocessed_text"] = df_no_nan["text"].apply(preprocess_text)
    corpus = df_no_nan["preprocessed_text"].tolist()

    # Create and save index
    model = BM25Okapi(corpus, tokenizer=nltk.word_tokenize)
    pkl_data = {
        "model": model,
        "data_index": df_no_nan.index,
    }
    with open(idxpath, "wb") as idx_file:
        pickle.dump(pkl_data, idx_file)
    return df_original
