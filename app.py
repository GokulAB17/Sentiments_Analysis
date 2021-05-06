
"""
Created on Sun Mar 14 20:51:06 2021

@author: Gokul A....
"""

# Importing the libraries
import pickle
import pandas as pd
import numpy as np
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "Sentiments_Analysis Using AI "

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")
    
def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    global balancedReviews
    global l1,l2,v1,v2,poscount,negcount
    
    
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    balancedReviews = pd.read_csv("balanced_review.csv")
    #{3: 527690, 2: 263866, 1: 263676, 4: 263676, 5: 263454}
    l1 = list(dict(balancedReviews['overall'].value_counts()).keys())
     #[2, 1, 4, 5]
    v1 = list(dict(balancedReviews['overall'].value_counts()).values())
    #[263866, 263676, 263676, 263454]
    
    l1= [str(i) for i in l1]
  
   # scrappedReviews.drop(["Unnamed: 0"], axis=1, inplace=True)
    balancedReviews.dropna(inplace = True)
    
    balancedReviews['overall'] != 3

    balancedReviews = balancedReviews[balancedReviews['overall'] != 3]

    balancedReviews['Positivity'] = np.where(balancedReviews['overall'] > 3, 1, 0 )
    balancedReviews['Positivity'].value_counts()
    l2 = list(dict(balancedReviews['Positivity'].value_counts()).keys())
     #[0, 1]
    v2 = list(dict(balancedReviews['Positivity'].value_counts()).values())
     #[527542, 527130]
    l2= [str(i) for i in l2]
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)
    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)

def check_review(reviewText):
    global vocab
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

def Scrapp_Reviews():
    global poscount,negcount
    poscount,negcount=0,0
    for i in range(len(scrappedReviews)):
        if check_review(scrappedReviews["reviews"][i])==1:
            poscount=poscount+1
       
        else:
            negcount=negcount+1
        
    return poscount,negcount
   
def create_app_ui():
    global project_name
    main_layout = dbc.Container(dbc.Jumbotron(
                [
                    html.H2(id = 'heading', children ="Sentiments Analysis Using AI Based Text Analytics Techniques" , className = 'display-6 mb-7'),
                    html.Br(),
                     html.Div([
                        html.Div([
                            dcc.RadioItems(
                            id='reviewtype',
                            options=[{'label': i, 'value': i} for i in ['balancedReviews', 'ScrappedReviews']],
                            value='balancedReviews',
                            labelStyle={'display': 'inline-block','margin-right': "20px"}
                            )
                            ],
                    style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(id='indicator-graphic')
        
                ]),
                     html.Br(),     
                    dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = scrappedReviews.reviews[0],
                    style = {'margin-bottom': '30px'}
                    
                )
                       ],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                        ),
                     html.Div(id = 'result'),
                     html.Br(),
               dbc.Container([     
                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = 'Project is working well', style = {'height': '100px'}),
                    dbc.Button("Check", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                      ],
                       style = {'padding-left': '50px', 'padding-right': '50px'}
                       ),
                     html.Div(id = 'result1'),
                    ],
                className = 'text-center'
                ),
        className = 'mt-4'
                        )
    return main_layout
@app.callback(
              Output('indicator-graphic', 'figure'),
                 [
                     Input('reviewtype', 'value')
                     ],
                 [
                     State('reviewtype', 'value')
                     ]
    
              
                )

def up_graph (reviewtype,value):                                                                                                         
    global scrappedReviews
    global balancedReviews
    global l1,l2,v1,v2,poscount,negcount 
    if reviewtype=="ScrappedReviews":
        labels = ['PositiveReviews','NegativeReviews']
        values = [poscount,negcount]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0,0.3])])
        fig.update_layout(title_text="Scrapped_Reviews from Etsy.com")
        return fig
    else:
        label1 =   l1
        label2 = ["Negative","Positive"] # Create subplots: use 'domain' type for Pie subplot
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=label1, values=v1, name="OverallReviews"),
              1, 1)
        fig.add_trace(go.Pie(labels=label2, values=v2, name="PolarisedReviews"),
              1, 2)

        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")

        fig.update_layout(
            title_text="Balanced_Reviews Sampled from Amazon Reviews Historical Data",
            # Add annotations in the center of the donut pies.
        annotations=[dict(text='OR', x=0.20, y=0.5, font_size=30, showarrow=False),
                     dict(text='PR', x=0.80, y=0.5, font_size=30, showarrow=False)])
    
        return fig 
@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result', 'children'),
    [
    Input('dropdown', 'value')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(dropdown, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative Review", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive Review", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
def main():
    global app
    global project_name
    global pickle_model
    global vocab
    global scrappedReviews
    global balancedReviews
    global l1,l2,v1,v2,poscount,negcount
    #print("Project Initiating.......")
    load_model()
    Scrapp_Reviews()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    #app = None
    #project_name = None
    #vocab = None
    #scrappedReviews = None
    #balancedReviews = None
    #l1,l2,v1,v2,poscount,negcount = None,None,None,None,None,None
    #print("Project Ended......")
if __name__ == '__main__':
    main()    
    

                            
