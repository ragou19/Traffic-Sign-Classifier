# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition CNN**

In this project, I build a convolutional neural network based on the **German Traffic Sign Database Benchmark (GTSDB)** dataset and use it to predict German road signs not present in the set. The goals/steps of this project are the following:
* Load the data set (see below for links to the project data set).
* Explore, summarize and visualize the data set.
* Design, train and test a model architecture.
* Use the model to make predictions on new images.
* Analyze the softmax probabilities of the new images.
* Summarize the results with a written report.


[//]: # (Image References)

[image1]: ./writeup_pics/signs_by_type.png "Traffic Signs by Type"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) assigned to this project individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ragou19/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Using numpy, summary statistics of the traffic signs data set were calculated:

* There are 34799 images in the training set.
* The validation set has 4410 images.
* The test set consists of 12630 images.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

An exploratory visualization of the data set lies below. It is a bar chart showing how the data is organized by sign type, the keys of which can be found in the [signnames.csv](https://github.com/ragou19/Traffic-Sign-Classifier/blob/master/signnames.csv) file contained in this repo. Examining this bar chart allows us to see roughly the relative distribution of images across types, which has an effect on the likelihood of the model choosing that image once it is developed:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I brightened all of the images in the dataset. Investigation of many of the images showed that a large number were indistinguishable due to extremely poor lighting conditions. Lightening the images proportional to their difference from max brightness was intended to bring out more of their latent features while not affecting already well-lit signs too much.

Next, the images were grayscaled because the color of the signs did not contribute to being able to distinguish them. All of the changes amongst the 43 classes of sign could be accounted for by differences in what shapes were printed on the signs.

Realizing that accounting for brightness in dark images was not enough, the contrast of each image was heightened by applying histogram equalization to the dataset. This converts an image so that an equal proportion of each of its expressed intensities is represented.

As a last step, I normalized the image data to allow pixellated values to take place in the region [-1,1], which allows for better processing with the model.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My convnet model consists of three convolution and pooling blended layers, followed by four fully connected layers, two ReLU layers between the first, second, and third fully connected layers, and finally a softmax layer.

Initially, my motivation for creating more layers than the LeNet architecture was to tailor my model for what I believed to be a more complicated dataset. Indeed, a dataset including colors and a larger variety of shapes would seem to require more layers. As I saw an improvement in accuracy after adding an additional fully connected layer, I felt no reason to revert from this decision, even after incorporating grayscale later into the problem.

While tinkering with the model, I found that ReLUs should not be incorporated in the last two fully connected layers, while dropout, for this particular problem, would unfortunately sacrifice accuracy too much for its benefit to model stability and reversing overfitting to be considered.

Here is a diagram of my final model and the layers it contains:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x9 	|
| Max pooling           | 2x2 stride, outputs 							|
| ReLU                  |                                               |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x27 	|
| Max pooling           | 2x2 stride, outputs 							|
| ReLU                  |                                               |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x81 	|
| Max pooling           | 2x2 stride, outputs 							|
| ReLU                  |                                               |
| Flatten               |                                              	|
| Fully connected		| outputs 194        							|
| ReLU                  |                                               |
| Fully connected		| outputs 116        							|
| ReLU                  |                                               |
| Fully connected		| outputs 71        							|
| Fully connected		| outputs 43        							|
| Softmax				| etc.        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adaptive Moment Estimation (Adam) optimizer with a batch size of 256 and epoch setting of 50. Running the model multiple times led me to choose the following values for my hyperparameters:

* Learning Rate: 0.0025
* Mu: 0
* Sigma: 0.04 / 128
* Regularization Beta: 0

An iterative process of changing hyperparameters one at a time was used to optimize each quantity in turn. The sigma was lessened by an integral factor in order to account for the normalization of the pixellated data described earlier.

Note that the beta for regularization is 0. As with other parameters in the model, activating a regularization beta had too big a negative effect on accuracy to have included it in the final version. This may have been remedied if more images were contained in the dataset - or created for it, which is an aspect of the problem yet awaiting exploration.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


