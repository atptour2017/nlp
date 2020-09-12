from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
tdm = vectorizer.fit_transform(corpus)
space = vectorizer.vocabulary_
print(space)