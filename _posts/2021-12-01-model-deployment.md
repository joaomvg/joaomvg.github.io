---
layout: post
title: "Model Deployment with AWS Sagemaker"
date: 2021-12-01
category: Cloud-Computing
image: 
excerpt: Deploy custom model in AWS Sagemaker using containers. 
katex: True
---

### Write custom Container

We can deploy our model in AWS Sagemaker using a custom container. We create a folder '/opt/program' inside the container where we store the files:
* serve: starts the server API
* predictor.py: defines Flask REST API

When Sagemaker runs the container it starts the CMD "serve", which deploys the REST API. The file
```bash
predictor.py
``` 
loads the pickled model and implements a Flask API with two methods that Sagemaker expects:
* [GET] /ping
* [POST] /invocations

The pickled model can be copied directly to the container to a folder of choice. Or it can be stored in a S3 bucket and passed on to Sagemaker as an artifact. Sagemaker then extracts the tar.gz file from S3 and copies it to the folder '/opt/ml/model'. Therefore, if we pass the model as an artifact, the predictor module needs to unpickle the file at '/opt/ml/model'.

The Dockerfile has the basic structure:
```Dockerfile
FROM ubuntu:latest

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         python3-pip\
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#Install python libraries
COPY requirements.txt /opt/program/
RUN python3 -m pip install /opt/prorgam/requirements.txt && \
        rm -rf /root/.cache

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

#copy model to /opt/ml/model or other folder
COPY model.pkl /opt/ml/model/
# Set up the program in the image
COPY model-files /opt/program
WORKDIR /opt/program
RUN chmod +x serve

CMD [ "serve" ]
```

We can run the container locally and test the API:
```bash
#build model
docker build -t sagemaker-model .
#run the container
docker run -p 8080:8080 sagemaker-model:latest 
```

Now we can access the API at 127.0.0.1:8080:
```bash
curl --location --request POST 'http://localhost:8080/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{"data": [[1,2],[3,4],[3,3],[10,1],[7,8]]}'
```

### Sagemaker Deployment

First we need to push our docker image to our AWS ECR repository. Assuming that we have already created a repository with URI: "aws_account_id".dkr.ecr."region".amazonaws.com/"name-model", we tag the docker image using the same repository URI, that is,

```bash
docker tag sagemaker-model:latest "aws_account_id".dkr.ecr."region".amazonaws.com/sagemaker-model:latest
```
and then push to the ECR repository (it presupposes that one has logged in)
```bash
docker push "aws_account_id".dkr.ecr."region".amazonaws.com/model-sagemaker:latest
```

Now that we have uploaded the docker image we can go to Sagemaker section and create a Model, an Endpoint Configuration and finaly deploy the model to an Endpoint.

#### **Create Model**
We give it a name
<div style="text-align: center"><img src="/blog-data-science/images/sagemaker_deployment_model_step.png"  width="80%"></div>
then we choose to "Provide model artifacts and image location" since we want to use our container
<div style="text-align: center"><img src="/blog-data-science/images/sagemaker_deployment_model_step2.png"  width="80%"></div>
and last we choose "single model" and then write the URI of the docker image. Since our container already has the pickled model we do not need to write anything in the box "Location of model artifacts"
<div style="text-align: center"><img src="/blog-data-science/images/sagemaker_deployment_model_step3.png"  width="80%"></div>

#### **Endpoint-Configuration**

We give it a name and then choose the model that we have created in previous step. At this point we need to choose the EC2 instance that will run the container.
<div style="text-align: center"><img src="/blog-data-science/images/sagemaker_deployement_endpntconfig.png"  width="80%"></div>

#### **Endpoint**

Give a name to the endpoint and then choose an existing endpoint-configuration, the one we have previously created:
<div style="text-align: center"><img src="/blog-data-science/images/sagemaker_deployment_endpoint.png"  width="80%"></div>
Then choose "Create Endpoint".

### Access Endpoint

Now that the model is deployed and the endpoint is in "Service", we build an API to call the container endpoint. There are essentially two ways of doing this:

1) We can invoke the Sagemaker endpoint directly. For this we need to create a role with permission to invoke the sagemaker endpoint.

2) Create a REST API Gateway with a Lambda to call the Sagemaker Endpoint.

#### 1. Invoke Sagemaker directly

In this case the AWS user must have the permission to invoke the sagemaker endpoint. Then we need the credentials **Access_Key_id** and **Secret_access_key** of this user. In Postman the request looks like
<div style="text-align: center"><img src="/blog-data-science/images/postman_access_endpoint.png"  width="100%"></div>