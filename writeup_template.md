# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_46_37_006.jpg "center"
[image3]: ./examples/center_2016_12_01_13_33_11_691.jpg "Recovery Image"
[image4]: ./examples/right_2016_12_01_13_46_37_006.jpg "Recovery Image"
[image5]: ./examples/left_2016_12_01_13_46_37_006.jpg "Recovery Image"
[image6]: ./examples/right_2016_12_01_13_46_37_006.jpg "Normal Image"
[image7]: ./examples/flip_right_2016_12_01_13_46_37_006.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report_final.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3X3 filter sizes and depths between 3 and 64. And after flattening we have total of 2112 inputs (model.py lines 60-101)

The model includes RELU layers to introduce nonlinearity (code line 65), and the data is normalized in the model using a Keras lambda layer (code line 60).


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 78, 88, 92, 96).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 117). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I did a validation split of the data for 20%.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and aldo different side driving.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to run the car in center of the lane and complete one circit in autonomus mode

My first step was to use a convolution neural network model similar to the Nvidia's deep learning seld driving cars network. I thought this model might be appropriate because:
 * It has 5 convolution layers which will help my model to separate the features more finely.
 * Also the 4 FC layers will increase in training speed
 * This model is well-tested model by Nvidia.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the Nvidia model so now we have different dropout layers at 4 places with the 50% dropout values.

Then I added Maxpooling layer in  4 of the convolution layers to add the reduce the parameter given and to be matched with Nvidia values.
and removed the subsampling which was there in Nvidia's network [https://developer.nvidia.com/blog/deep-learning-self-driving-cars/]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track mostly on turns, to improve the driving behavior in these cases, I used the data for recovering from the left and right sides

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 60-101) consisted of a convolution neural network with the following layers and layer sizes ,
This model contain 5 conv2D layes and 4 dense layers and with one flatten layer this will combine to total 9 layers.
Here is the summary of the my model

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)           (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 66, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 62, 316, 24)       1824
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 158, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 27, 154, 36)       21636
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 77, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 73, 48)        43248
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 37, 48)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 35, 64)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 33, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================

```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from both the direction of the road. These images show what a recovery looks like starting from one left side and right side:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data set, I also flipped images and angles thinking that this would increase the data from the other direction of the road as track is a circular one which will give angles for left side only. If we flip the images and angles it will give the data as if we are driving in exact mirror of the loop. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 48216 number of data points. I then preprocessed this data by first normalization by lambda function and cropping Cropping2D(cropping=((70, 24), (0, 0)))



I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the data below but I tried running my training sets for different epochs and took 10 as the max number of epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

```sh

38572/38572 [==============================] - 94s 2ms/step - loss: 0.0285 - val_loss: 0.0224
Epoch 2/10
38572/38572 [==============================] - 89s 2ms/step - loss: 0.0243 - val_loss: 0.0210
Epoch 3/10
38572/38572 [==============================] - 90s 2ms/step - loss: 0.0232 - val_loss: 0.0217
Epoch 4/10
38572/38572 [==============================] - 89s 2ms/step - loss: 0.0225 - val_loss: 0.0218
Epoch 5/10
38572/38572 [==============================] - 90s 2ms/step - loss: 0.0219 - val_loss: 0.0202
Epoch 6/10
38572/38572 [==============================] - 90s 2ms/step - loss: 0.0215 - val_loss: 0.0197
Epoch 7/10
38572/38572 [==============================] - 90s 2ms/step - loss: 0.0214 - val_loss: 0.0207
Epoch 8/10
38572/38572 [==============================] - 89s 2ms/step - loss: 0.0209 - val_loss: 0.0219
Epoch 9/10
38572/38572 [==============================] - 90s 2ms/step - loss: 0.0210 - val_loss: 0.0202
```
