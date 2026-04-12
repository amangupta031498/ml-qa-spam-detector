
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_home():
  res = requests.get(BASE_URL)
  assert res.status_code == 200

def test_valid_prediction():
  res = requests.post(f"{BASE_URL}/predict", params={"text" : "Free money!!!"})
  assert "prediction" in res.json()

def test_empty_input():
  res = requests.post(f"{BASE_URL}/predict",params={"text":""} )
  assert res.status_code == 400
