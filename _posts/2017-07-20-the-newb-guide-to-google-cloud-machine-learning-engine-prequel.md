+---
+title: The newb guide to Google Cloud Machine Learning (ML) Engine - Prequel
+author: Michael Yee
+published: false
+---

If you are new to Google Cloud Platform and its products, you might find the documentation a bit lacking or tough to interpret without concrete examples.  You could be a Google search wizard and can easily find whatever resources you need to learn from.  As for myself, finding an easy step-by-step resource is like looking for a needle in a haystack.

This blog is my journey on how to implement Google Could Machine Learning (ML) Engine.  You are expected to have some Python and TensorFlow background to be able to flow this guide.  

virtualenv
A virtual environment is a tool to keep the dependencies required by different projects in separate places, by creating virtual Python environments for them.  To learn more about virtual environment, you may checkout this url:  http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/

  $ pip install virtualenv											

Creating virtualenv
I will make a project folder called ml-engine and in that folder, I will create a virtual environment call ml-venv using the following command:

  $ virtualenv -p /usr/bin/python3.5 ml-venv 								

Starting and stopping the virtualenv
To start using the virtual environment, it needs to be activated:

  $ source ml-venv/bin/activate 										
  (ml-venv) $													

To start using the virtual environment, it needs to be activated:

  (ml-venv) $ deactivate											
  $ 														

TensorFlow
TensorFlow ( https://www.tensorflow.org/) is an open source software library for numerical computation using data flow graphs, developed by Google for machine learning. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. TensorFlow’s architecture allows the flexible computational deployment to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 

  (ml-venv)  $ pip install  tensorflow==1.2.1

Google Cloud storage
...

Google Cloud Machine Learning Engine
...

Cloud SDK
Command-line interface for Google Cloud Platform products and services (https://cloud.google.com/sdk/downloads#apt-get).  This package contains the gcloud, gcloud alpha, gcloud beta, gsutil, and bq commands only. It does not include kubectl or the App Engine extensions required to deploy an application using gcloud commands. 

Create an environment variable for the correct distribution: 

  (ml-venv)  $export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"			

Add the Cloud SDK distribution URI as a package source: 

  (ml-venv)  $echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.lis

Import the Google Cloud public key: 

  (ml-venv)  $curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

Update and install the Cloud SDK: 

  (ml-venv)  $sudo apt-get update && sudo apt-get install google-cloud-sdk				

Install the additional components: 

  (ml-venv)  $sudo apt-get install google-cloud-sdk-app-engine-python			

Run gcloud init to get started:

  (ml-venv)  $gcloud init		
