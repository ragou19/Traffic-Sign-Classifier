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
[image2]: ./pictures/5.jpg "Traffic Sign 1"
[image3]: ./pictures/7.jpg "Traffic Sign 2"
[image4]: ./pictures/8.jpg "Traffic Sign 3"
[image5]: ./pictures/32.jpg "Traffic Sign 4"
[image6]: ./pictures/33.jpg "Traffic Sign 5"
[image7]: ./pictures/38.jpg "Traffic Sign 6"

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

Initially, my motivation for creating more layers than the LeNet architecture was to tailor my model for what I believed to be a more complicated dataset. Indeed, a dataset including colors and a larger variety of shapes seemed to require more layers. As I saw an improvement in accuracy after adding an additional fully connected layer, I felt no reason to revert from this decision, even after incorporating grayscale later into the problem.

While tinkering with the model, I found that ReLUs should not be incorporated in the last two fully connected layers, while dropout, for this particular problem, would unfortunately sacrifice accuracy too much for its benefit to model stability and reversing overfitting to be considered.

Here is a diagram of my final model and the layers it contains:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x9 	|
| Max pooling           | 2x2 stride, outputs 							|
| ReLU                  | extends model space & reduces sparsity		|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x27 	|
| Max pooling           | 2x2 stride, outputs 							|
| ReLU                  | extends model space & reduces sparsity		|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x81 	|
| Max pooling           | 2x2 stride, outputs 							|
| ReLU                  | extends model space & reduces sparsity		|
| Flatten               | make 1D output for further processing 		|
| Fully connected		| outputs 194        							|
| ReLU                  | extends model space & reduces sparsity		|
| Fully connected		| outputs 116        							|
| ReLU                  | extends model space & reduces sparsity		|
| Fully connected		| outputs 71        							|
| Fully connected		| outputs 43        							|
| Softmax				| converts outputs into probabilities			|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adaptive Moment Estimation (Adam) optimizer with a batch size of 256 and epoch setting of 50. Running the model multiple times led me to choose the following values for my hyperparameters:

* Learning Rate: 0.0025
* Mu: 0
* Sigma: 0.04 / 128
* Regularization Beta: 0

An iterative process of changing hyperparameters one at a time was used to optimize each quantity in turn. The sigma was lessened by an integral factor in order to account for the normalization of the pixellated data described earlier.

Note that the beta for regularization is 0. As with other parameters in the model, activating a regularization beta had too big a negative effect on accuracy to have included it in the final version. This may have been remedied if more images were contained in the dataset - or created for it, which is an aspect of the problem yet awaiting exploration.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


From the training, my final results were:

* training set accuracy of 99.4%.
* validation set accuracy of 94.4%
* test set accuracy of 91.2%

In order to achieve this accuracy, I began with the LeNet archtecture, then inserted an extra layer into the neural network to improve accuracy and allow for more complex modeling. Rampant underfitting led me to believe that the LeNet architecture was not well-equipped enough to handle the particular type of data being considered. Adding another fully connected layer improved accuracy across sets of hyperparameters, most likely due to the network's ability to condense higher-level features more smoothly than with simply three such layers. Significant gains were not made again by adding even more layers of either convolution or fully connected type in addition to the previous layer.

Next, each parameter was fine-tuned in turn until optimal performance was achieved across rounds of tuning. As mentioned above, factors such as learning rate, regularization beta, dropout, ReLU count, ReLU position, the mean and standard deviation of the weight distribution, and number of epochs and batches were toggled in a round-robin fashion.

Quite literally, a running log of changes were made in the form of comments in the block where the validation, test, and training accuracies were calculated (shortly after defining the neural network layers). This log consisted of steps where either one or a small amount of parameters were edited once each time, and then the network rerun to gauge its relative effect on accuracy. 

Curiously enough, the effect of dropout and regularization was undesirable to this model, as each caused accuracy to drop below acceptable thresholds. Even dramatically enlarging the number of epochs and changing the batch size was not enough to compensate for the these losses. For instance, an introduction of 90% dropout to any given layer would cause final accuraciesto dip 5% or more on each test, which was unacceptable. Regularization, in the same way as dropout, caused there to be a tighter margin between training and validation accuracies reported, but led to an overall decrease of validation accuracy.

Enlarging the test set would have been a good way to be able to introduce dropout and regularization for the purposes of smoothing the accuracy progression and lessening overfitting.

One last technique employed relied on tailoring the learning rate dependent on the performance of the network. This "datarate wrangling" consisted of dramatically decreasing it when the validation accuracy lied above the desired 93% threshold, and then increasing it by a similar ratio if it ever dipped below. By doing so over successive applications, I was able to constrain decreases around values above 93% most of the time, with the validation accuracy ultimately settling at 94.4%.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]	![alt text][image7]

The images overall, while being mostly exceptionally clear and unadultered examples, may be difficult to classify simply because of their novelty. As suggested earlier, overfitting in the dataset due to lack of examples driving down the ability to reliably incorporate dropout and regularization meant that these web images may be misclassified anyway into other sign types exhibiting similar characteristics. This did, in fact, turn out to be the case, which we shall see later.

It is also evident that the images occur with differing amounts of smoothness and sharpness, and it is difficult to realize or predict how much the network may depend on smoothness with regards to classification.

It can be safely said that patterns emerged with regards to these classifications; namely, across multiple runs, those whose predictions turned out false (in accordance with the next question) were more frequently to be among the the three that seemed more blurry (images 32, 33, and 5).


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| End of all limits		| End of all speed and passing limits (same)	| 
| Turn right ahead		| Turn right ahead								|
| Keep right			| Keep right									|
| 80 km/h	      		| Vehicles over 3.5 metric tons prohibited		|
| 100 km/h				| 100 km/h 		    							|
| 120 km/h 				| 120 km/h 										|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. This compares fairly well with the accuracy on the test set, which was 91.2%.

#### 3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

For the first image, the model is sure that this is an "end of all speed and passing limits" sign than any other type (probability of ), and the image does in fact contain an "end of all limits" sign. The top five softmax probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .222         			| End of all speed and passing limits 			| 
| .210     				| Go straight or right 							|
| .208					| Priority road									|
| .196	      			| Ahead only					 				|
| .165				    | Turn right ahead     							|


For the second image, the model concluded with a 24.9% probability that we had the correct type of sign, a "turn right ahead" sign. Other probabilities were as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .249         			| Turn right ahead 								| 
| .204     				| Road work 									|
| .199					| Ahead only									|
| .197	      			| Go straight or right			 				|
| .151				    | End of all speed and passing limits    		|


Third, the "keep right" sign was predicted correctly with a probability of 28.5%, as shown below along with the other top 5 softmax probabilities for this image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .285         			| Keep right	 								| 
| .202     				| Turn left ahead 								|
| .193					| Priority road									|
| .175	      			| Vehicles over 3.5 metric tons prohibited		|
| .145				    | Go straight or left 					   		|


The fourth sign, an 80 km/h speed limit sign, was predicted incorrectly as a "vehicles over 3.5 metric tons prohibited" sign, but with comparative probabilities of 23.9% and 22.2%, came pretty close to being correct:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .239         			| Vehicles over 3.5 metric tons prohibited		| 
| .222     				| No passing	 								|
| .216					| Speed limit (60 km/h)							|
| .168	      			| Keep right									|
| .155				    | End of no passing 					   		|

For the fifth sign, the model correctly predicted it as a 100 km/h speed limit sign with a probability of 22.5%. The other softmax probabilities were as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .225         			| Speed limit (100 km/h)						| 
| .217     				| Speed limit (50 km/h)							|
| .194					| No vehicles									|
| .185	      			| Speed limit (30 km/h)							|
| .179				    | Speed limit (60 km/h) 				   		|


It is interesting to note how many speed limit signs came up in the top 5 probabilities for this particular speed limit sign.

Lastly, the 120 km/h speed limit sign was correctly predicted with a probability of 27.6%. The top 5 softmax probabilities for this sign were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .276         			| Speed limit (120 km/h)						| 
| .206     				| Speed limit (20 km/h)							|
| .176					| Dangerous curve to the left					|
| .173	      			| Dangerous curve to the right					|
| .169				    | Speed limit (70 km/h) 				   		|


Again, many different speed limit signs were predicted in the top 5 probabilities for this speed limit sign.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Looking at the visualization of the first convolution layer of the network upon our first web-based image, we see that the model has decided to focus on a variety of elementary characteristics such as edge shading, relative contrast, detection of edge, and material characteristics (smoothness of the sign vs. roughness of the background foliage).

The following is a table summarizing characteristics present in each of the feature maps that result, as I have interpreted them:

| Map	| Characteristics Present in Feature Map						| 
|:-----:|:-------------------------------------------------------------:| 
| 0    	| Gradients of edges toward the top left of the image			| 
| 1 	| Contrast or smoothness										|
| 2		| Raisedness of foreground texture information					|
| 3		| Indentation of background texture information					|
| 4		| Random interesting pixel-based characteristics 		   		|
| 5		| Brightness 											   		|
| 6		| Blend detection between shapes/sign and external shapes  		|
| 7		| Gradients of edges toward the top of the image 				|
| 8		| Detection of textures toward the bottom of the image 			|

Unfortunately, I was unable to resolve the bug just above the images in the notebook, where a certain variable somewhere is evaluated as "NoneType" on a subsequent iteration of the loop. However, this had no apparent effect on the feature maps in question, which all printed as evinced by a previous check on the range of the featuremaps variable in the last code block.