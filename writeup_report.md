# **Behavioral Cloning** 

## Writeup for Behavioral clonning project

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/recoveryLeft.jpg "Recovery Image"
[image2]: ./examples/recoveryRight.jpg "Recovery Image"
[image3]: ./examples/architecture.png "Model architecture"
[video1]: ./examples/video.mp4 "Output video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model and model is quite readable.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I chose NVIDIA's self driving car architecture provided in the lessons since it provided better result compared to LeNet.
This architecture has normalizing layer fallowed by 5 convolutional layer fallowed by 4 fully connected layers. Please go through the model.py for more detail. 

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting, I reduced the number of epochs to 10 as I observed after 10 epochs the validation loss was increasing which might be because of overfitting. Also I dropout of 0.2(20%) in the model provided good improvement on validation loss.     

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section


#### 5. Solution Design Approach

I have tried using LeNet's architecture first but the the result with NVIDIA's self driveing car model were far better. Hence I chose NVIDIA's self driving car architecture provided in the lessons.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set as the epochs increased. This implied that the model was overfitting. Hence I added the dropout layer and limited the number of epochs to 10. It provded good results on training the model.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I took recorded recovery path from left and right sides of the road. After feeding the recovery data along with centre driving data, the model was observed to perform well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 6. Final Model Architecture

This NVIDA's self driving car architecture which I used has normalizing layer fallowed by 5 convolutional layer fallowed by 4 fully connected layers. 
Here is a visualization of the architecture.

![alt text][image3]


#### 7. Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

I find using simulator to gather training data bit difficult as probaby I am not good at video games but I tried my best to get the training data which includes center driving laps, center driving lap in opposite direction and recovery paths. However I observe that, there were a few spots where the vehicle fell off the track, for which I recorded recovery path data for those particular portions. Hence the final combination of training and validation data is center lane driving, recovering from the left and right sides of the road.

These images show what a recovery looks :

![alt text][image1]
![alt text][image2]

After collecting data, I then preprocessed this data by using lambda function to normalization and mean center the data. After that I cropped the image to remove sky and car hood from the data set. 

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by training process. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 8. Final result of the Model 

Please find the video.mp4 showing the result of how the Car was able to drive well on the test data and it stayed on the tracks.  
![alt text][video1]