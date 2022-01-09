---
layout: post
title: "Deploying with Docker Containers"
date: 2021-08-30
category: Cloud-Computing
image: container_cover.png
excerpt: I explain how to use Docker containers to deploy a machine learning application.
katex: True
---

### Dockerfile

The Dockerfile contains a set of instructions which are used to build a new Docker image. The starting point of these instructions is a base image. Then other commands follow like copying files, installing dependencies or running programs. 

The purpose of a docker image is to run a self contained and lightweight version of an operating system- the Docker container. This makes it very useful to deploy machine learning models because we do not need to worry about the operating system of the host nor we need to setup the environment (like Python) every time we need to deploy a new model.

For example, the following Dockerfile deploys a machine learning model via a Flask application that we wrote in the file app.py:
```Dockerfile
FROM python:3.8.2-slim

# Copy function code
COPY app.py rf_model.pkl /app/

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target /app/

#run api
CMD python3 /app/app.py
```
The build starts from the base image "python:3.8.2-slim", copies the app.py program and the saved machine learning model (random forest object) into the folder "/app/", and installs necessary dependencies. When the image is run as a container the line with the "CMD" keyword is run by default- this is our machine learning API.

First we build this image with name "app", and then run the container
```shell
docker build -t app .
docker run -p 5001:5000 app
```
The flag "-p 5001:5000" forwards container port 5000 to port 5001 on the localhost. We can now call the API at localhost:5001:
```shell
curl --location --request POST 'http://127.0.0.1:5001/model/invoke' 
--header 'Content-Type: application/json' 
--data-raw '{"data":[[6.49517312e-01, -1.63477913e+00,  1.02223807e+00, -2.90998418e-01,
        4.08584955e-01, -2.51346205e-01, -1.19300836e+00, -7.79194513e-02,
        1.89090598e-04,  1.43111208e+00, -1.58314852e+00,  1.67256137e+00,
       -2.12077154e+00]]}'
```
From which we get the answer:
```shell
{
    "prediction": [
        41.089999999999996
    ]
}
```