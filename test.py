import requests
import pandas as pd
from pprint import pprint

df = pd.read_csv('Gathered data.csv')

data = df.iloc[1].to_json()

# print(data)
response = requests.post("https://test-recommend.herokuapp.com/predict",
                         json={"survey": data.replace('\\', '')})
print(response.text)