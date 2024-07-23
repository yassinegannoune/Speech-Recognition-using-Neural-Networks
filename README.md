# Speech-Recognition-using-Neural-Networks

## Overview

This project was executed using Google Colab. Due to the limitations of the free version of Colab, which provides GPU services for a limited amount of time, we divided the code into three separate files. This approach was taken to efficiently manage training processes and time computation.

## Files and Directories

### Training Files

- The project contains three separate training files instead of a single comprehensive file. This segmentation allows for better utilization of the available GPU time in Colab's free tier.

### Results Folder

- The `results` folder contains all the binary files in `.h5` format. These files store the trained models. Instead of retraining the models from scratch, you can directly test the models using these files.

## Running the Code on Your Local Machine

To run this code on your PC, follow these steps:

1. **Change the Path**:
   - Modify the path of LibriSpeech to point to your personal drive and name it `LibriSpeech`.

2. **Convert Files to `.wav`**:
   - Ensure all files in your `LibriSpeech` directory are converted to `.wav` format using the `code_converter.py` script.

3. **Install Necessary Packages**:
   - Install all the required packages specified in the project.

## Authors

Best regards from the authors.
if you have any question reach out:yassine.gannoune@usmba.ac.ma

linkedin:linkedin.com/in/yassine-gannoune-5a7405228
