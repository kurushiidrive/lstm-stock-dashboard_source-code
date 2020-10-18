# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 12:53:54 2020

@author: kurushiidrive
"""

import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

external_stylesheet = ['https://raw.githubusercontent.com/kurushiidrive/lstm-stock-prediction-app/master/dash_stylesheet.css']

app = dash.Dash(external_stylesheets=external_stylesheet)
server = app.server

@server.route("/")
def blah():
    print('blah')
    return 0

SPLIT = 6732

scaler = MinMaxScaler(feature_range=(0,1))

my_style = {
    'textAlign': 'center',
    'font-family': 'Avenir, Helvetica, sans-serif'    
}

df = pd.read_csv('https://raw.githubusercontent.com/kurushiidrive/lstm-stock-prediction-app/master/datasets_541298_1054465_stocks_AMD.csv')
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

all_close = new_data.values
valid_close = all_close[SPLIT:, :]

scaled_close = scaler.fit_transform(all_close)

model = load_model("/app/amd_model")

input_data = new_data[len(new_data) - len(valid_close) - 60:].values
input_data = input_data.reshape(-1, 1)
input_data = scaler.transform(input_data)

X_test = []
for i in range(60, input_data.shape[0]):
    X_test.append(input_data[i-60:i-1, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_closing_price = model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)



valid = new_data[SPLIT:]
train = new_data[:SPLIT]
valid['Predictions'] = predicted_closing_price

# app
app.layout = html.Div(style={'backgroundColor': '#3fb0ac'}, children=[
    
    html.H1('Stock Price Analysis Dashboard', style=my_style),
    
    dcc.Tabs(id='tabs', style=my_style, children=[
        
        dcc.Tab(label='Advanced Micro Devices (NASDAQ: AMD)', 
                style=my_style, 
                children=[
            html.Div([
                html.H2('LSTM Closing Price Predictions w/ Actual Closing Price', style=my_style),
                
                dcc.Checklist(
                    id='hide-or-show_graph',
                    options=[
                        {'label': 'Training', 'value': 'TR'},
                        {'label': 'Prediction', 'value': 'PR'},
                        {'label': 'Actual', 'value': 'AC'}
                    ],
                    value=['TR', 'PR', 'AC'],
                    labelStyle={'display': 'inline-block'},
                    style={'backgroundColor': '#FFFFFF',  'textAlign': 'center', 'font-family': 'Avenir, Helvetica, sans-serif', 'padding-top': '25px'}
                ),
                
                dcc.Graph(
                    id='LSTM',
                    figure={
                        'data':[
                            go.Scatter(
                                x=train.index,
                                y=train['Close'],
                                mode='lines',
                                name='Training'
                                ),
                            go.Scatter(
                                x=valid.index,
                                y=valid['Predictions'],
                                mode='lines',
                                name='Prediction'
                                ),
                            go.Scatter(
                                x=valid.index,
                                y=valid['Close'],
                                mode='lines',
                                name='Actual'
                                )
                            ],
                        'layout':go.Layout(
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Price (USD)'}
                            )
                        }
                    ),
                html.H2('Historic Stock Data (17 Mar 1980 to 1 Apr 2020)', style=my_style),
                dcc.Graph(
                    id='Historic Data',
                    figure={
                        'data':[
                            go.Scatter(
                                x=new_data.index,
                                y=new_data['Close'],
                                mode='lines',
                                text=data["Open"],
                                customdata=data["Volume"],
                                name='',
                                hovertemplate=
                                '<i>Open</i>: %{text:.2f}<br>' +
                                '<i>Close</i>: %{y:.2f}<br>' +
                                '<i>Volume</i>: %{customdata}'
                                )
                            ],
                        'layout':go.Layout(
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Price (USD)'}
                            )
                        }
                    )
                ])
            
            ]) # Tab 1
        
        ]) # Tabs
    
    ]) # app.layout

@app.callback(
    Output(component_id='LSTM', component_property='figure'),
    [Input(component_id='hide-or-show_graph', component_property='value')],
    [State(component_id='LSTM', component_property='figure')])
def hideorshow (tags, fig_state):
    fig_state['data'][0]['visible'] = 'TR' in tags
    fig_state['data'][1]['visible'] = 'PR' in tags
    fig_state['data'][2]['visible'] = 'AC' in tags
    return fig_state


if __name__ == '__main__':
    app.run_server(debug=False)