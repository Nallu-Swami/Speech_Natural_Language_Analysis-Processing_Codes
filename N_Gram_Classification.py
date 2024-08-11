import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'], compression='zip')
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Using unigrams (single words)
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test_vect)
print("Accuracy without n-grams:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Using n-grams (unigrams + bigrams)
vectorizer_ngram = CountVectorizer(ngram_range=(1, 2))  # Unigrams and bigrams
X_train_vect_ngram = vectorizer_ngram.fit_transform(X_train)
X_test_vect_ngram = vectorizer_ngram.transform(X_test)

# Train a Logistic Regression model
model_ngram = LogisticRegression(max_iter=1000)
model_ngram.fit(X_train_vect_ngram, y_train)

# Predict and Evaluate
y_pred_ngram = model_ngram.predict(X_test_vect_ngram)
print("Accuracy with n-grams:", accuracy_score(y_test, y_pred_ngram))
print(classification_report(y_test, y_pred_ngram))



