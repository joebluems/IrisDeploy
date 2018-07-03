import pandas as pd
import pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

#Load the model
clf = 'iris_svm_456.sav'
loaded_model = None
with open('./model/'+clf,'rb') as f:
  loaded_model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def apicall():
  """API Call Pandas dataframe (sent as a payload) from API Call """
  try:
    test_json = request.get_json()
    test = pd.read_json(test_json, orient='records')
    test = test[['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']]
    score = test.drop(columns=['class'])
  except Exception as e:
    print e
	
  if score.empty:
    return(bad_request())

  predictions = loaded_model.predict(score)
  prediction_series = list(pd.Series(predictions))
  cols = ['actual','SVM_result']
  final_predictions = pd.DataFrame(list(zip(test['class'], prediction_series)),columns=cols)
  responses = jsonify(predictions=final_predictions.to_json(orient="records"))
  responses.status_code = 200

  return (responses)
