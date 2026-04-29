import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("data/spam_ham_dataset.csv")
assert 'label' in df.columns
assert 'text' in df.columns
assert df.isnull().sum().sum() == 0
df['label'] = df['label'].map({'ham' : 0, 'spam': 1})
X_train,X_test, y_train , y_test = train_test_split(df['text'], df['label'], test_size = 0.2, random_state = 42, stratify = df['label'] )
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,y_train)

train_preds = model.predict(X_train_vec)
test_preds = model.predict(X_test_vec)

train_acc = accuracy_score(y_train,train_preds)
test_acc = accuracy_score(y_test,test_preds)

print(f"Training accuracy: {train_acc}")
print(f"Test accuracy: {test_acc}")
print(classification_report(y_test, test_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_preds))
assert acc > 0.85, "model accuracy too low"
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
print("model evaluated, trained and saved")
