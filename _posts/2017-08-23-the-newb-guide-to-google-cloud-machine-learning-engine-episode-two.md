---
title: The newb guide to Google Cloud Machine Learning (ML) Engine - Episode Two
author: Michael Yee
published: true
---

I have mixed feeling about cliffhangers and I hope you do not hate me for leaving you with one in the last blog.  In this episode, we will look at the last piece of the puzzle for predictions using Cloud Machine Learning Engine REST APIs.

## Answer Part 3:  Cloud Machine Learning Engine REST APIs

The following code describes how to use the Google API client library to easily make calls to the Cloud Machine Learning Engine REST APIs and is designed to familiarize yourself with with the following ideas:

- Use application default credentials in your Python applications
- Get a Python representation of the Cloud ML Engine services
- Use that representation to create a model in your project

```python
import json

from googleapiclient import discovery
from googleapiclient import errors
from oauth2client.client import GoogleCredentials
from tensorflow.python.lib.io import file_io


# Google Storage locations
BUCKET                  = "my-ml-project-888"
GS_BUCKET               = "gs://my-ml-project-888/"
INPUT_FOLDER            = "input/"
OUTPUT_FOLDER           = "output/"
MODEL_FOLDER            = "regression"
INPUT_FILENAME          = "ML_engine_feed_dict.json"
OUTPUT_FILENAME         = "ML_engine_result.json"


def read_bucket_json(filename):
    # reads a json file from Google Storage
    with file_io.FileIO(GS_BUCKET+INPUT_FOLDER+filename, "r") as json_file:
        json_data = json.load(json_file)

    return json_data


def write_bucket_json(json_data, filename):
    # writes a json file to Google Storage
    with file_io.FileIO(GS_BUCKET+OUTPUT_FOLDER+filename, "w") as json_file:
        json.dump(json_data, json_file)


# store the project ID as a variable in the format the API requires
projectID = "projects/{}/models/{}".format(BUCKET, MODEL_FOLDER)

# get application default credentials
credentials = GoogleCredentials.get_application_default()

# build a representation of the Cloud ML API
ml = discovery.build("ml", "v1", credentials=credentials)

# read the input file from Google Storage
feed_dict = read_bucket_json(INPUT_FILENAME)

# create a dictionary with the fields from the request body
request_dict = {"instances": [feed_dict]}

# create a request to call projects.models.create
request = ml.projects().predict(name=projectID, body=request_dict)

try:
    # make the prediction request!
    response = request.execute()
    
    # print the result to screen
    print(response)

    # write the result to Google Storage
    write_bucket_json(response, OUTPUT_FILENAME)

except errors.HttpError as err:
    # something went wrong!
    print("There was an error!  Here are the details:")
    print(err._get_reason())

```

The above code will make a request to Google Cloud Machine Learning (ML) Engine using data and model located on Google Storage.  

Input instances (data) are passed for online prediction as the message body for the projects.predict call. Make each instance an item in a list and name the list member instances.  Example: {"instances": [{'x': [1.0, 2.0, 3.0]}]}

The output will print the result {'predictions': [{'y': [1.5, 2.0, 2.5]}]} to the screen and to an output file located on Google Storage.  

# The End?