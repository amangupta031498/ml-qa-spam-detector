
import pickle
def test_model_loaded():
  model = pickle.load(open("model/model.pkl", "rb"))
  assert model is not None

def test_prediction_output():
  model = pickle.load(open("model/model.pkl", "rb"))
  vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

  sample = ["Win money now!!"]
  vec = vectorizer.transform(sample)
  pred = model.predict(vec)
  assert pred[0] in [0,1]
