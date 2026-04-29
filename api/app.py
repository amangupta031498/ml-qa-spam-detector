
from fastapi import FastAPI, HTTPException
import pickle
import time

app = FastAPI()

# Assign the loaded model and vectorizer to global variables
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.get("/")
def home():
  return{"message" : "Spam detection APi running"}

@app.post("/predict")
def predict(text:str):
  if not text or text.strip() == "":
    raise HTTPException(status_code = 400, detail = "Empty input not allowed")

  start_time = time.time()
  vec = vectorizer.transform([text])
  prediction = model.predict(vec)[0]
  response_time = time.time() - start_time
  label = "Spam" if prediction ==1 else "Ham"

  return{
      "prediction" : label,
      "response_time": response_time
  }
