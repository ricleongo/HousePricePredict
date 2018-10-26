## How Jupyter Notebook connects with API Web Service:
So, for this project, Sklearn library was used to first create a multi linear regression model, and then to trained it from the data we were exploring and analysing from Census API and Zillow CSV, through Jupyter Notebook as it is explained in `data_exploration.ipynb` file.

Then using the same library sklearn, it was created a file from the model using `sklearn.externals.joblib` and finally it was saved inside back-end folder as `capita_model.model`.

> With the model in place inside the folder where is located the API Web Service we are able to consume this model.

Finally `IPython.display.HTML` library was used to dynamically create HTML content in Jupyter Notebook, so that way it was created `model_test.html` file which is the one that shows the web page. 

Inside it has JQuery language in order to interact with buttons and controls inside the HTML and then be able to hit our Web service, that should be under this route: `http://localhost:5000/LinearRegressionPredict`



### Installing API Web Service:
In order to run the API Web Service locally we first need to install Flask (web microframework for python) locally, so to do that please open Terminal or Command Line and type the next command:

> (sudo) pip install Flask

or

> conda install -c anaconda flask


### Running the API Web Service locally:
After Flask is installed then you are able to run the Web Service API with these two commands:

> export FLASK_APP=back-end/web_api.py

> flask run --host=0.0.0.0

After this you are able to connect Jupyter Notebook with this API Web Service