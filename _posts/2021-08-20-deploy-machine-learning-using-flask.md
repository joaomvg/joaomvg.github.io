---
layout: post
title: "Deploy Machine Learning using Flask"
date: 2021-08-20 09:20
image: consoleapi.jpg
categories: Cloud-Computing
---

In this post I explain how to run a machine learning model on a cloud computer and access its predictions via an API.

To do this goal I write a simple API interface using Flask in Python. I have trained a linear regression model on artificial data and saved the model object using Pickle. The code below loads the model and runs an API that receives POST requests for the model predictions.

```python
from flask import Flask, jsonify, abort,request
import numpy as np
import pickle

app = Flask(__name__)

lr=pickle.load(open('lr_model.pkl','rb'))

@app.route('/')
def index():
    return "Linear Regression model API."

@app.route('/model/invoke',methods=['POST'])
def model():
    if not request.json or not 'data' in request.json:
        abort(400)
    data=request.json['data']
    x=np.array(data)
    pred=lr.predict(x)

    return jsonify({'prediction': pred.tolist()}), 201

if __name__ == '__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')

```
To test the api, run this script on a remote cloud instance (Linode offers up to 100 dollars to experiment their services). Then allow for incoming connections on port 5000, as set in the script (or any other port of your choice).  Send a POST request to the api using the IP address of the instance, for example

```python
curl --location --request POST 'xxx.xxx.xxx.xxx:5000/model/invoke' \
--header 'Content-Type: application/json' \
--data-raw '{"data":[2,3,4,5]}'
```
On the remote, you can see a POST request from the Ip address of your local computer:
![](/blog-data-science/images/consoleapi.jpg)

And this is the response from the api call:
```python
{
  "prediction": [
    4.354176603044118, 
    6.384373814367889, 
    8.414571025691659, 
    10.44476823701543
  ]
}

```
