#%%
from flask import request
from flask import Flask, jsonify
from decorator import crossdomain
from sklearn.externals import joblib
import pandas as pd
import time

app = Flask(__name__)

@app.route('/healtcheck')
def healthcheck():
    response_message =  {
        'name': 'API',
        'version': '1.0.0'
    }

    return jsonify(response_message)

@app.route('/hello', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def hello():
    message = request.get_json(force=True)
    name = message['name']
    return f'Hello {name}'


@app.route('/LinearRegressionPredict', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def LinearRegressionPredict():
    message = request.get_json(force=True)

    county = message['county']
    percapita = message['percapita']

    value = pd.DataFrame.from_dict([{ "MunicipalCodeFIPS": county, "Per Capita Income": percapita }])

    model = joblib.load("./back-end/capita_model.model")
    response = model.predict(value)

    response = ["{0:,.2f}".format(x) for x in response]
    
    return f'Price predicted based on per capita income: {response}'
