# Classification-Algo-using-Pytorch
An object classification model built using Pytorch to detect quality of tomatoes. Dataset was taken from www.kaggle.com, and model was trained in google colab.

-----------------------------------------------------------------------
I was not able to upload the data set as the file was too large. If you want the exact set of data with which i am working.
Here is the link to -  [Data Set](https://www.kaggle.com/datasets/sumn2u/riped-and-unriped-tomato-dataset).

## NOTE 

Make sure to change the path of the data set in the script.
-----------------------------------------------------------------------

## General Steps we are following (Just an overview)

1. Problem Statement Hypothetical Business Situation: Tomato Classification Model Background You are working as a data scientist for "FreshHarvest Inc.," a leading agricultural technology company that specializes in providing innovative solutions to improve crop yield and quality. The company has recently partnered with several large tomato farms to help them automate the sorting process of ripe and unripe tomatoes. The goal is to build an accurate and efficient classification model to distinguish between ripe and unripe tomatoes, ensuring that only the best quality produce reaches the market. Business Need Tomato sorting is currently done manually, which is time-consuming, labor-intensive, and prone to human error. FreshHarvest Inc. aims to implement an automated system that uses computer vision and machine learning to classify tomatoes based on ripeness. This system will reduce labor costs, increase sorting speed, and improve the overall quality of tomatoes sent to market.

2. Data Collection Kaggle - 177 images and labels of mixed riped and unriped tomatos.

3. Data Wrangling Remove Corrupted Images Ensure label accuracy Image Quality

4. Exploratory Data Analysis Distribution of Classes: Check the distribution of ripe and unripe tomatoes to ensure there is no significant class imbalance.

5. PreProcessing Data Augmentation: rotation, flipping, zooming, and shifting to artificially increase the diversity of the dataset Normalization: Normalize the pixel values of the images to a range of [0, 1] to improve model training stability.

6. Build and Train model CNN

7. Test model Binary Cross Entropy

8. Deploy! model.save_dict()


-----------------------------------------------------------------------

## Technical Steps:

Step 1: We have to create a dataset -> dataloader
It's important to preprocess the data

Step 2: visualize the images with it's corresponding labels.

Step 3: Create a model
- You either choose a model (If you are a beginner, you should choose a pre exisiting model.)
- Or Build a model

Step 4: Depending on overfitting / underfitting
- Tweak the learnable parameters.
  
Step 5: Use Integral attribution to explain features.

-----------------------------------------------------------------------

### GOAL 
My goal is to reach above 90% accuracy with my model.

### GOAL ACHIEVED
My model was able to achieve 83% accuracy.
