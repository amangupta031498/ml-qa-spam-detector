
import pandas as pd

def test_no_null():
  df = pd.read_csv("data/spam_ham_dataset.csv")
  assert df.isnull().sum().sum() ==0
def test_label_values():
  df = pd.read_csv("data/spam_ham_dataset.csv")
  assert set(df['label'].unique()) == {'ham', 'spam'}
