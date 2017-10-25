---
title: Automating ML Training With Jenkins Pipelines
author: Mike Kroutikov
layout: post
---

What I want is:

1. Train models in Amazon EC2 cloud. GPU instances there are quite cheap
2. Automatically shutdown cloud worker(s) when training is done. This avoids paying for idle machine.
3. Store training data and experiment results in Google Storage. This is dictated by our framework choice (TensorFlow). TF
   natively works with Google Storage urls, but does not yet support S3 urls. In more details:
   * use worker local disk as `temp` space for downloading dataset in its native format, and unpacking.
     We have to use local storage because tools like `tar`, `zip`, `curl` only work with local filesystem, no cloud.
   * use Google Storage bucket to store preprocessed data. TF has built-in unitilies for data preparation
     and storing in space-efficient `TFRecord` format. All these utilities transparently support Google Storage cloud
     urls (but do not support S3 urls yet). This data is durable, and we will be training many models off it. Therefore
     we can not use worker local storage - data must survive worker termination!
   * use Google Storage bucket to store experiment results - checkpoints and TesorBoard events. Again, this data has
     to be durable. Since TensorBoard natively supports Google Storage urls, we can visualise experiment results
     right off the Google Storage bucket!

Doing the above manually is not hard, but does not scale well. Manually provisioning and starting EC2 workers
is tedious and repetitive. If my training is expected to complete in the middle of the night, who will stop the EC2 machine?

Another concern is managing credentials. We generally need:

1. Private keys to connect to EC2 workers
2. Username/password to checkout private repositories from GitHub and/or Bitbucket
3. Private keys to enable upload to Google Storage bucket
4. Secret configuration files to access private PyPI

If I need to launch a new clean instance, I would need to move secrets to it somehow.

How to orchestrate all this with minimal manual effort and cost? Answer is... Jenkins! Read on.

## Jenkins to rule us all!
[Jenkins](https://jenkins.io/) is one of the best and most widely used integration tools out there.

Yes, it looks aging and cranky, and occasionally dies with OOM error (thanks, Java!). But it has an enourmous
number of useful plugins. And it is free.

Recent addition of Piplines in Jenkins is a very welcome development: now I can store job descriptions
in the source tree and have proper versioning and change history.

Global configuration (users, credentials, plugins) is still manual though. But I will live with that for the time being.
In any case, I do not want to put secret keys in source control.

## The Plan

1. Use Jenkin's [**Amazon EC2 Plugin**](https://wiki.jenkins.io/display/JENKINS/Amazon+EC2+Plugin)
   and configure it to launch the AMI we want. We should assign a good label to the cloud EC2 Jenkins
   workers - plugin will launch those on-demand when job requires a worker with that specific label.
2. Create Google Cloud service account with the appropriate scope (`Google Storage Admin` role) and download its JSON
   secret file. Name it `gcloud-jenkins.json`. We need this to read/write from/to Google Storage.
3. In Jenkins, configure a credential of type *Secret File*, name it `gcloud-secret-file` and 
   upload `gcloud-jenkins.json` file.
4. In Jenkins, configure *Username/Password* credentials to access SCM (say, GitHub and/or Bitbucket). We will use it to
   checkout private repositories.
5. In Jenkins, configure a credential of type *Secret File*, name it `pip-conf` and upload private PIP config. 
   This will enable us to access private PyPI repositories

## The Solution

Here is a Jenkins pipeline that does the thing ([gist](https://gist.github.com/mkroutikov/19ec3e0efd43a21ca93b7a5e6b4672f7)):
```groovy
pipeline {
  
  // use only nodes marked as 'tensorflow'
  agent { node { label 'tensorflow' } }
  
  // build parameters - these are prompted interactively
  parameters {
    string(defaultValue: '', description: 'Problem Name', name: 'problem')
  }
  
  environment {
    // convenience: define params as env variables
    PROBLEM_NAME = "${params.problem}"
    BUCKET       = "gs://training.innodatalabs.com/${params.problem}"
  }
  
  stages {
      
    // make sure our private PyPI is accessible from this node
    stage('Provision Private PyPI') {
      steps {
        withCredentials([file(credentialsId: "pip-conf-secret-file", variable: 'PIP_CONF')]) {
          sh "mkdir -p ~/.config/pip; cp -f $PIP_CONF ~/.config/pip/pip.conf"
        }
      }
    }
        
    // apt install all required packages
    // EC2 comes up with apt update processes already running. Therefore have to wait up to 10 minutes
    // before our apt install can succeed
    stage('Provision virtualenv') {
      steps {
        retry(20) {
          sleep(30)
          sh 'sudo apt-get install virtualenv -y'
        }
      }
    }

    // check out project and prepare Python3 virtual environment
    stage('Prepare') {
      steps {
        git credentialsId: "mikes-github-username-password", url: 'https://github.com/mkroutikov/my-cool-private-repo.git', branch: 'master'
        sh 'rm -rf .venv; virtualenv .venv -p python3'
        sh '''
        . .venv/bin/activate
        pip install -r requirements.txt
        pip install tensorflow
        pip install -e .
        '''
      }
    }
        
    // do the real thing. Since tensorflow trainer writes to Google Storage, need
    // GOOGLE_APLICATION_CREDENTIALS. For completeness, add timeout
    stage('Training') {
      steps {
        echo "Training problem $PROBLEM_NAME"
        withCredentials([file(credentialsId: "gcloud-storage-secret", variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
          timeout(time: 5, unit: 'HOURS') {
            sh """
            . .venv/bin/activate
            my-cool-trainer --data_dir $BUCKET/data --problem $PROBLEM_NAME
            """
          }
        }
        echo "All done"
      }
    }
  }
}
```

Well, that is a mouthful, for sure. 

## The Explanation
Let us look at each piece separately.

### Declare the agent
```groovy
agent { node { label 'tensorflow' } }
```
We want this job to be executed on a worker machine having label `tensorflow`.

I will **not** allocate any permanent machine as a `tensorflow` worker though. Instead,
I will configure Amazon EC2 plugin to spin off EC2 worker on-demand, and stop it after 10 minute idle timeout.

Configuring Amazon EC2 plugin is straihhtforward. Following are the important points:
1. Choose "Ubuntu Deep Learning" public AMI image. It comes with CUDA libraries installed.
2. Cap number of instances to avoid Jenkins spinning up too many workers.
3. Choose the option to Stop instance on idle timeout. Otherwise it will be terminating idle instance. 
   Spinning up stopped instance is significantly faster.
4. By default, worker policy is *Use this node as much as possible*. Change it to 
   *Use this worker only for jobs with matching labels*. These machines are expensive and we do not want to 
   spin them up for anything else.
5. Use *Advanced* menu to add tags: Name=managed-by-jenkins. This helps me to see what is going on when I look at AWS EC2
   console.
6. Add label `tensorflow` to the configured AMI worker.

### Prompt for parameters
My training job is parametrized (naturally). I am using `parameters` block to declare variables that training needs.
At the build time Jenkins will prompt user for the values.

```groovy
parameters {
  string(defaultValue: '', description: 'Problem Name', name: 'problem')
}
```

In any step I can now refer to the parameter as ${params.problem}. More realistic training job will have
few more parameters: `model`, `hparam`, etc.

### Set-up the Environment
For convenience I define some environment variables. Note that previously declared parameters can be used
when building variable value.

```groovy
environment {
  PROBLEM_NAME = "${params.problem}"
  BUCKET = "gs://training.innodatalabs.com/${params.problem}"
}
```

### Provision access to private PyPI
On my laptop I have a file `~/.config/pip/pip.conf` that adds private PyPI repository. This way `pip install`
transparently works with public packages and private packages.

To have the same facility on EC2 worker I will configure "Secret File" in Jenkins. Then, provision step looks like this:
```groovy
stage('Provision Private PyPI') {
 steps {
   withCredentials([file(credentialsId: "pip-conf-secret-file", variable: 'PIP_CONF')]) {
     sh "mkdir -p ~/.config/pip; cp -f $PIP_CONF ~/.config/pip/pip.conf"
   }
 }
}
```
Most interesting part here is `withCredentials` arument. Note the name `pip-conf-secret-file`. It refers to a credentials configured in Jenkins. 

To configure this I go to `Jenkins/Credentials` menu and further choose `System` sub-menu. Then `Add credentials`.
Choose credential type to be `Secret File`, enter `pip-conf-secret-file` as credentials id, and upload my `pip.conf`.

### Provision Packages
Then we want to make sure that packages we need are installed. Specifically, I will need `virtualenv` one.

That would be as simple as running
```bash
apt update; apt install virtualenv -y
```

But... That does not work on the newly started EC2! 

The reason being that newly created instances automatically run updates at instance creation time, in the background. 
Instance may seem ready to work, but some `apt` process(es) are running in background, keeping `apt lock`. Attempt to
run `apt install` will result in the error aquiring the lock.

We need to wait for the background updates to complete before installing our packages.

My solution is to keep trying for about 10 minuts. Typically, automatic updates will complete in about 5-6 minutes.

Here is the Pipeline part for that:
```groovy
stage('Provision packages') {
 steps {
   retry(20) {
     sleep(30)
     sh 'sudo apt-get install virtualenv -y'
   }
 }
}
```

Now we are done with the general provisioning. Time to think about doing the real stuff.

### Prepare for work
In this step I will check out the repository, create virtual environment, and install project dependencies with pip.

```groovy
stage('Prepare') {
 steps {
   sh 'rm -rf .venv; virtualenv .venv -p python3'
   git credentialsId: "mikes-github-username-password", url: 'https://github.com/mkroutikov/my-cool-private-repo.git', branch: 'master'
   sh '''
   . .venv/bin/activate
   pip install -r requirements.txt
   pip install tensorflow
   pip install -e .
   '''
 }
}
```
I start with removing virtual environment created by the previous build, and creating a new fresh one. 
If I re-use old virtual envirnoment, I can save 1-2 minutes by not installing all from scratch.
But I would rather have 100% reproducible build and take this time/cost hit.

Also note that I am not using `--system-site-packages` flag when creating virtual environment. This will ignore any
packages pre-installed globally in the image. One of them is system-wide installed `tensorflow-gpu`. I want
to follow the best practices and have full control over the python package versions.

Next, I am checking out my private repository from HitGub. Note the familiar technique of supplying credentials
by its name. You should have guessed by now that these credentials were configured in Jenkins as "Username/Password" 
credentials with my name and password.

Things to note in the preparation step are:
* I am installing tensorflow explicitly. This is because it is not in my `requirements.txt`
* I am using development install of my repository. The reason for this is cosmetic: I have `console_scripts` command
  defined in `setup.py` and want to use it as an executable command (without the need to run the `python` explicitly).
  The result of running this development install will be that command `my-cool-trainer` is now available in the
  virtual environment!

### Doing the work
  
```groovy
stage('Training') {
 steps {
   withCredentials([file(credentialsId: "gclud-storage-secret-file", variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
     timeout(time: 5, unit: 'HOURS') {
       sh """
       . .venv/bin/activate
       my-cool-trainer --data_dir $BUCKET/data --problem $PROBLEM_NAME
       """
     }
   }
   echo "All done"
 }
}
```
This all should be very familiar now. The body of the stag is wrapped in `withCredentials` block. Here we use it
to expose environment variable `$GOOGLE_APPLICATION_CREDENTUIALS`, pointing to a secret file
containing Google service account keys in JSON format. We need this to give EC2 worker read/write acces to Google Storage
bucket.

Step body is wrapped in `timeout` block. This is to control training time. With some hyperparameters choice
training may run forever.

## Conclusion

What I can do now is: 

1. Trigger my training from Jenkins UI.
2. Start many jobs that will either run sequentially, or in parallel on multiple EC2 workers (this is controlled
   by the instance cap we set when configuring Amazon EC2 Plugin). If not enough workers are available
   training job will stay in the Jenkins queue waiting for the next available worker.
3. Monitor training progress with TesorBoard tool, pointing it to the Google Storage bucket with experiments.
4. Last, but not least - I can stop worrying about EC2 workers idling and wasting my budget.
