import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_word2vec_classifier(documents, labels):
    """
    Trains a Word2Vec + Logistic Regression classifier on the given text data.

    Args:
        documents (list of str): List of document texts.
        labels (list): List of labels corresponding to each document.

    Returns:
        model: Trained Logistic Regression classifier.
        accuracy: Accuracy on test split.
    """

    tokenized_docs = [
        re.sub(r"[^\w\s]", "", doc.lower()).split()
        for doc in documents
    ]

    w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

    def document_vector(doc):
        words = [word for word in doc if word in w2v_model.wv.key_to_index]
        if not words:
            return np.zeros(w2v_model.vector_size)
        return np.mean([w2v_model.wv[word] for word in words], axis=0)

    doc_vectors = np.array([document_vector(doc) for doc in tokenized_docs])

    clf = LogisticRegression(max_iter=50)
    clf.fit(doc_vectors, labels)
    preds = clf.predict(doc_vectors)
    print("Accuracy (on training data):", accuracy_score(labels, preds))

    return clf,(accuracy_score(labels, preds))
