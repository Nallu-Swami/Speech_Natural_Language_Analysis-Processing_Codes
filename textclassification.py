import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ssl
#try using this SSL enabler if getting error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

data = {
    'text': [
        "Free money now!!!",
        "Call me at 555-1234",
        "Win a free trip to Bahamas",
        "Are we meeting tomorrow?",
        "You have won a lottery, claim your prize now!",
        "Let's have lunch tomorrow",
        "Congratulations! You've been selected for a prize",
        "Hi, how are you?",
        "Your loan is approved, apply now!",
        "Don't forget the meeting at 10 AM."
    ],
    'label': [
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

def preprocess(text, use_stemming=True, remove_stopwords=True):
    words = text.lower().split()
    if remove_stopwords:
        words = [word for word in words if word not in stop_words]
    if use_stemming:
        words = [ps.stem(word) for word in words]
    return ' '.join(words)

X_train_processed = X_train.apply(lambda x: preprocess(x))
X_test_processed = X_test.apply(lambda x: preprocess(x))

# Vectorization (Convert text data into numerical features)
vectorizer = CountVectorizer()

# Without preprocessing
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# With preprocessing
X_train_processed_vect = vectorizer.fit_transform(X_train_processed)
X_test_processed_vect = vectorizer.transform(X_test_processed)

# Train a classifier (Logistic Regression)

# Without preprocessing
model = LogisticRegression()
model.fit(X_train_vect, y_train)
y_pred = model.predict(X_test_vect)
print("Accuracy without preprocessing:", accuracy_score(y_test, y_pred))

# With preprocessing
model_processed = LogisticRegression()
model_processed.fit(X_train_processed_vect, y_train)
y_pred_processed = model_processed.predict(X_test_processed_vect)
print("Accuracy with preprocessing:", accuracy_score(y_test, y_pred_processed))
