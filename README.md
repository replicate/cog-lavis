# Cog template for mPLUG-Owl


[![Replicate](https://replicate.com/replicate/blip2-instruct-vicuna13b/badge)](https://replicate.com/joehoover/blip2-instruct-vicuna13b)

This repo provides an unofficial implementation of `InstructBLIP Vicuna13B` for replicate. 

For more details, please refer to the original [paper](http://arxiv.org/abs/2305.06500) and Github [repository](https://github.com/replicate/cog-lavis/tree/main/projects/instructblip).



## Prerequisites

- **GPU machine**. You'll need a Linux machine with an NVIDIA GPU attached and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. If you don't already have access to a machine with a GPU, check out our [guide to getting a 
GPU machine](https://replicate.com/docs/guides/get-a-gpu-machine).

- **Docker**. You'll be using the [Cog](https://github.com/replicate/cog) command-line tool to build and push a model. Cog uses Docker to create containers for models.


## Step 1: Install Cog

First, [install Cog](https://github.com/replicate/cog#install):

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

## Step 2: Run the model

You can run the model locally to test it:

```
cog predict -i img=@docs/_static/Confusing-Pictures.jpg -i prompt="What is unusual about this image?" -i
```

## Step 3: Create a model on Replicate

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model.

Make sure to specify "private" to keep the model private.

## Step 5: Configure the model to run on A100 GPUs

Replicate supports running models on a variety of GPUs. The default GPU type is a T4, but for best performance you'll want to configure your model to run on an A100.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 6: Push the model to Replicate

Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 3:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)

# Usage and License 

The model is intended and licensed for research use only. InstructBLIP w/ Vicuna models are restricted to uses that follow the license agreement of LLaMA and Vicuna. The models have been trained on the LLaVA dataset which is CC BY NC 4.0 (allowing only non-commercial use).