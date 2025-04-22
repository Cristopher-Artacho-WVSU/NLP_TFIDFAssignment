from term_frequency import compute_tf
from tf_idf import compute_idf, compute_tfidf
from cosine_similarity import cosine_similarity
from word2vec_classifier import train_word2vec_classifier

import wikipediaapi
wiki = wikipediaapi.Wikipedia('language=en')

topics = ["Python programming language", "C++ programming language", "Java programming language", "C# programming language", "C programming language"]
documents = []

for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        documents.append(page.summary)
    else:
        print(f"Page '{topic}' not found.")

# Sample corpus
# documents = [
#     "the cat sat on the mat", # considered as a document, document 1
#     "the dog sat on the log", # document 2
#     "cats and dogs are great pets" # document 3
# ]


print("Documents:")
for doc in documents:
    print(doc)

# Tokenize and apply lowercase the documents into words
tokenized_docs = [doc.lower().split() for doc in documents]

# Create a set of unique words (vocabulary)
vocabulary = set(word for doc in tokenized_docs for word in doc)

# Compute the term frequency for each document
tf_vectors =  [ compute_tf(doc, vocabulary) for doc in tokenized_docs ]

print("\nTerm Frequency Vectors:")
for i, tf_vector in enumerate(tf_vectors):
    print(f"Document {i+1}: {tf_vector}")

# Compute the Inverse Document Frequency (IDF)
idf = compute_idf(tokenized_docs, vocabulary)
print("\nInverse Document Frequency:")
for term, idf_value in idf.items():
    print(f"{term}: {idf_value}")

tfidf_vectors = [ compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors ]
print("\nTF-IDF Vectors:") 
for i, tfidf_vector in enumerate(tfidf_vectors):
    print(f"Document {i+1}: {tfidf_vector}")

# Compute the Cosine Similarity between the first two documents
print("\nCosine Similarity Between All Document Pairs:")
num_docs = len(tfidf_vectors)
for i in range(num_docs):
    for j in range(i + 1, num_docs):  # Avoid duplicate and self comparisons
        sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j], vocabulary)
        print(f"Similarity between Document {i+1} and Document {j+1}: {sim}")


labels = [1, 2, 3, 4, 5]
model, acc = train_word2vec_classifier(documents, labels)

acc = train_word2vec_classifier(documents, labels)
print("Accuracy:", acc)