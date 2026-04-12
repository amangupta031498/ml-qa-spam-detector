import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("data/spam_ham_dataset.csv")
assert 'label' in df.columns
assert 'text' in df.columns
assert df.isnull().sum().sum() == 0
df['label'] = df['label'].map({'ham' : 0, 'spam': 1})
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = MultinomialNB()
model.fit(X,y)

preds = model.predict(X)
acc = accuracy_score(y,preds)
print(f"Training accuracy: {acc}")
assert acc > 0.85, "model accuracy too low"
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
print("model trained and saved")
