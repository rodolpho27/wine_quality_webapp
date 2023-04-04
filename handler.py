import pandas as pd 
import pickle 
from flask import Flask, request
from wine_quality.WineQuality import WineQuality 
import os

#load model 
model = pickle.load(open('model/model_winequality.pkl', 'rb'))

#instance Flask 
app = Flask( __name__ )

@app.route('/predict', methods=['POST'])
def predict():
	test_json = request.get_json()

	#collect data
	if test_json:
		if isinstance(test_json, dict):
			df_raw = pd.DataFrame(test_json, index=[0])
		else:
			df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

	#instance data preparation
	pipeline = WineQuality()

	#data preparation
	pipeline.data_preparation(df_raw)


	#prediction
	pred = model.predict(df_raw)

	#response
	df_raw['predicition'] = pred 

	return df_raw.to_json(orient='records')

if __name__ == '__main__':
	port = os.environ.get('PORT', 5000)
	app.run(host='0.0.0.0', port=port)