# LSTM Stock Dashboard Source Code

This repository holds the source code for my stock analysis web app hosted on [Heroku](https://kurushiid-stock-dashboard.herokuapp.com). The app itself was written in pure Python, using the Dash framework to build a fully interactive application with few lines of Python code.

In the dashboard, one can view select stocks' closing prices over time alongside their LSTM-predicted trend lines so as to understand the full contextual stability of a stock, and make well-informed decisions on whether and when to invest.

The app uses the Keras library for deep learning to extract predictions from models trained on their respective stocks. The code to train the models can be found [here](https://github.com/kurushiidrive/lstm-stock-dashboard), where I am hosting supplementary files used by the app, as well as the Github Pages embedding. Each model takes an input of the last X closing prices of its corresponding stock, in order to make a prediction of what the next closing price will be. For example, the AMD model uses the last 59 closing prices to make a prediction on the next one.

The normaliser used on the data was Sklearn's MinMaxScaler.

The full list of Python dependencies for this project's development can be found in this repository's `requirements.txt` file, but in short:
	- keras 2.3.1
	- dash
	- sklearn
	- pandas
	- numpy
	- tensorflow 1.14
	- gunicorn (for deployment)

`app.py` is the only source file in this repository, whereas the rest of the files are either serialisation files (`amd_model` and such) or files required for deployment to Heroku (`Procfile`, `requirements.txt`, `.gitignore`, etc.).
