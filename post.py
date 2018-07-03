import pandas as pd
import json
import requests
from pprint import pprint


#### Setting the headers to send and accept json responses ####
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

#### Reading data & sampling then convert to JSON #### 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv('./iris.csv', names=names)
df = df.sample(frac=0.1,replace=False, random_state=1)
data = df.to_json(orient='records')

##### MODEL # 1 KNN from Python ####
resp = requests.post("http://jblue1:5000/predict", data = json.dumps(data),headers= header)
print("Results from sklearn KNN model API...")
x = resp.json()
y=json.loads(x['predictions'])
for item in y:
  print("KNN: %s		Actual: %s" % (item['KNN_result'],item['actual']))

#### MODEL # 2 SVM from Python  ####
resp = requests.post("http://jblue1:5001/predict", data = json.dumps(data),headers= header)
print("\nResults from sklearn SVM model API...")
x = resp.json()
y=json.loads(x['predictions'])
for item in y:
  print("SVM: %s		Actual: %s" % (item['SVM_result'],item['actual']))


##### MODEL # 3 RF from R  #########
print("\nResults from Random Forest (R) model API...")
for index, row in df.iterrows():
   d = '{"Sepal.Length":%.1f, "Sepal.Width":%.1f,"Petal.Length":%.1f,"Petal.Width":%.1f}' \
      % (row["sepal-length"],row["sepal-width"],row["petal-length"],row["petal-width"])
   resp = requests.post("http://jblue1:5002/iris_api", data = d,headers= header)
   print("RF:  %s		Actual:%s" % (resp.text,row['class']))

