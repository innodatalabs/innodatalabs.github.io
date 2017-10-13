---
title: The newb guide to Google Cloud Machine Learning (ML) Engine - Epilogue
author: Michael Yee
published: true
---

In this eplilogue, I will descibe how to convert a saved model (checkpoint version) to a Freeze Graph (protobuf version) or a SavedModel (Google Cloud Machine Learning (ML) Engine format).

## Foreword

When you are programming with TensorFlow, you are defining a structured solution by describing and not computing its operations. All these operations are organized as a Graph and the results of these operations are Tensors. 

For computations to occur, the graph must be launched in a Session. When a Session is created, a new scope is started for the program where operations and its resulting Tensors are processed. Once the Session has ended, the Graph returns to its initial static state.

Summary

Graph 
- Describes the flow control and mathematical operations 
- Restoring or saving equates to loading/saving the Graph, metadata and variables
- The Graph requires a proper input pipeline (i.e. feed_dict, Dataset API, etc.)

Session and evaluation
- Variables get initialized and operations are executed

NOTE: Only variables will persist between multiple session as all other Tensors are temporary

## Part 1: Saving and restoring model (checkpoint version)

The Saver class provides methods to save and restore operations and variables between different sessions as checkpoints.  The following is an example of saving a model:

```python

import os
import tensorflow as tf


def model():
    folder = os.path.dirname(os.path.realpath(__file__))

    # start session
    session = tf.Session()

    # start model build
    # model variables 
    m = tf.Variable(0.5, name="m")
    b = tf.Variable(1.0, name="b")

    # feed_dict placeholder 
    x = tf.placeholder(tf.float32, name='x')
    feed_dict = {"x:0": 2.0}

    # operation 
    y = tf.add(tf.multiply(m, x), b, name="y")
    # end model build

    # initialize all the variables
    session.run(tf.global_variables_initializer())

    # run the operation
    print(session.run(y, feed_dict=feed_dict))
    # output is 2.0 which is m*x+b

    # saver object 
    saver = tf.train.Saver() 

    # saving the default graph and variables m and b 
    saver.save(session, os.path.join(folder,'model'))


def main(_):
    print('Running model...')
    model()


if __name__ == '__main__':
    tf.app.run()

```

In the above example, the Saver class created four files: 

- checkpoint: A list of all the saved checkpoint filenames
- model.data-00000-of-00001: weight data
- model.index: tensor name-data location table
- model.meta: Graph and metadata

The following is an example of restoring a model using import_meta_graph:

```python

import os
import tensorflow as tf


folder = os.path.dirname(os.path.realpath(__file__))

# create feed_dict to feed new data
feed_dict = {"x:0": 4.0}

# restore the meta checkpoint 
saver = tf.train.import_meta_graph('model.meta')

# access the default graph
graph = tf.get_default_graph()

# access the tensors to restore
m = graph.get_tensor_by_name("m:0")
b = graph.get_tensor_by_name("b:0")

# access the placeholder to restore
x = graph.get_tensor_by_name("x:0")

# access the operation to restore
y = graph.get_tensor_by_name("y:0")

with tf.Session() as session:
    # restore weights
    saver.restore(session, os.path.join(folder,'model'))

    # run the operation
    print(session.run(y, feed_dict=feed_dict)) 
    # output is 3.0 which is m*x+b

```

When the Saver class restores a meta checkpoint, the saved graph and associated metadata is loaded into the current default Graph.  

NOTE:  Weights only exists within a Session and to restore weights inside a Graph, it must be done within a Session.

## Part 2: Freeze Graph

It is often not very convenient to have separate files when you are deploying to production.  If only there was a way to package everything nicely into one file to facilitate storage, updates and versioning.  Luckily in TensorFlow, there is the freeze_graph.py script that takes a set of checkpoints and freezes them together into a single file.

From the freeze_graph.py script, I have "borrowed" certain bits of the code to highlight the important parts for our discussion.

```python

import tensorflow as tf
from tensorflow.python.framework import graph_util


# filename for the output graph
output_graph_filename = "frozen_graph.pb"

# remove all explicit device specifications
clear_devices = True

# output node(s) to save
output_node_names = "y"

# restore the meta checkpoint 
saver = tf.train.import_meta_graph(meta_graph_or_file='model.meta', clear_devices=clear_devices)

# access the default graph
graph = tf.get_default_graph()

# retrieve the protobuf graph definition
input_graph_def = graph.as_graph_def()

with tf.Session() as session:
    # restore weights
    saver.restore(session, 'model')

    # TensorFlow built-in helper to export variables to constants
    output_graph_def = graph_util.convert_variables_to_constants(
        sess=session,
        input_graph_def=input_graph_def, # GraphDef object holding the network
        output_node_names=output_node_names.split(",") # List of name strings for the result nodes of the graph
    ) 

    # serialize and dump the output graph
    with tf.gfile.GFile(output_graph_filename, "wb") as f:
        f.write(output_graph_def.SerializeToString())

```

In the above code, it has exported the Graph to a single self-sufficient file called "frozen_graph.pb".

Now, how does one use this frozen graph?  I'm glad you that you asked.  In the following example, we will...

1) import the graph_def ProtoBuf
2) load the graph_def into the current default Graph
3) access the input and output nodes 
4) run a Session

```python

import tensorflow as tf


# frozen graph filename
frozen_graph_filename = "frozen_graph.pb"

# create feed_dict to feed new data
feed_dict = {"x:0": 6.0}

# load the protobuf file
with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    #parse the data to retrieve the unserialized graph_def
    graph_def.ParseFromString(f.read())

# import a graph_def into the current default Graph
with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        graph_def=graph_def, 
        input_map=None, 
        return_elements=None, 
        name="",
        op_dict=None, 
        producer_op_list=None)

    # access the placeholder to restore
    x = graph.get_tensor_by_name("x:0")

    # access the operation to restore
    y = graph.get_tensor_by_name("y:0")

    with tf.Session() as session:
        # run the operation
        print(session.run(y, feed_dict=feed_dict)) 
        # output is 4.0 which is m*x+b

```

Note: When loading the frozen Graph, all operations would have gotten prefixed by "import" due to the parameter “name”, if left None, in the “import_graph_def” function.

## Part 3: SavedModel

To convert checkpoint files to a format that Google Cloud Machine Learning (ML) Engine accepts, all it is needed is some extra information around the Graph. Assuming the graph does not require assets (i.e. vocabularies), all that is needed is a serving signature.

```python

import tensorflow as tf
from tensorflow.python.framework import graph_util


# folder to export SavedModel
SavedModel_folder = "SavedModel"

# remove all explicit device specifications
clear_devices = True

# builds the SavedModel protocol buffer and saves variables and assets
builder = tf.saved_model.builder.SavedModelBuilder(SavedModel_folder)

# map of signature defs to be added to the meta graph def
signature_def_map = {}

# restore the meta checkpoint 
saver = tf.train.import_meta_graph(meta_graph_or_file='model.meta', clear_devices=clear_devices)

# access the default graph
graph = tf.get_default_graph()

# access the placeholder to restore
x = graph.get_tensor_by_name("x:0")

# access the operation to restore
y = graph.get_tensor_by_name("y:0")

with tf.Session() as session:
    # restore weights
    saver.restore(session, 'model')

    signature_def_map[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
    tf.saved_model.signature_def_utils.predict_signature_def(
        {"inputs": x}, {"ouputs": y})
    
    # adds the current meta graph and saves variables to the SavedModel
    builder.add_meta_graph_and_variables(sess=session,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_def_map)

    # writes a SavedModel protocol buffer to disk
    builder.save()

```

In the above example, the SavedModelBuilder class created a folder named "SavedModel" and all the files necessary for Google Cloud Machine Learning (ML) Engine is within this folder. I have uploaded the contents of this folder to a Google Storage location named "converted" and created the corresponding model on Google Cloud Machine Learning (ML) Engine.

The following code describes how to use the Google API client library to easily make calls to the Google Cloud Machine Learning (ML) Engine REST APIs.

```python

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors


# Google Storage locations
BUCKET                  = "my-ml-project-888"
MODEL_FOLDER            = "converted"

# create feed_dict to feed new data
feed_dict = {"inputs": 6.0}

# store the project ID as a variable in the format the API requires
projectID = 'projects/{}/models/{}'.format(BUCKET, MODEL_FOLDER)

# get application default credentials
credentials = GoogleCredentials.get_application_default()

# build a representation of the Cloud ML API
ml = discovery.build('ml', 'v1', credentials=credentials)

# create a dictionary with the fields from the request body
request_dict = {'instances': [feed_dict]}

# create a request to call projects.models.create
request = ml.projects().predict(name=projectID, body=request_dict)

try:
    # make the prediction request!
    response = request.execute()
    
    # print the result to screen
    print(response)

except errors.HttpError as err:
    # something went wrong!
    print("There was an error!  Here are the details: {}".format(err._get_reason()))

```

The above code will make a request to Google Cloud Machine Learning (ML) Engine using the model located on Google Storage.  

Input instances (data) are passed for online prediction as the message body for the projects.predict call. In our example, we have passed {"inputs": 6.0} as our data for online prediction.

The output will print the result {'predictions': [{'ouputs': 4.0}]} 
to the screen.  
