A deep learning model developed to classify tomatoes as either ripe or unripe using computer vision techniques. The project was inspired by the need to automate the sorting process in agricultural industries, particularly in the context of “FreshHarvest Inc.,” a hypothetical company that aims to improve crop yield and quality through advanced technology. The model leverages a Convolutional Neural Network (CNN) and employs techniques such as transfer learning using ResNet50, data augmentation, and integrated gradients for model interpretability. The model’s training process is monitored using TensorBoard to visualize training and validation losses, helping to ensure optimal model performance. Dataset was taken from www.kaggle.com, and model was trained in google colab.

**At a bigger scale it is entirely feasible to integrate this tomato classification model with an AI-driven machine (Robots), leveraging OpenCV for real-time image processing and control.**

-- I could have made a prototype for this but to be honest I cant afford buying materials to build an robot arm. --

-----------------------------------------------------------------------

<img src="https://github.com/user-attachments/assets/9b8a25a4-6c43-428a-86c5-be83a97f235f" alt="drawing" style="width:550px;"/>

-----------------------------------------------------------------------

Outcome

(losses started relatively high and decrease over time, this indicates the model was learning and improving its predictions).

![image](https://github.com/user-attachments/assets/68d0f9c2-a6f7-4825-b768-71458982c9a1)

-----------------------------------------------------------------------

grayscale usuage (It helps us to simplify an image by reducing the complexity of color information, which is an important part of this project).

**Formula used - Gray = 0.2989 × R + 0.5870 × G + 0.1140 × B**

![image](https://github.com/user-attachments/assets/121fd512-634a-415b-9324-0882f552ef97)
![image](https://github.com/user-attachments/assets/179f8edb-85dc-46e4-8a05-ab45267f1ba0)

-----------------------------------------------------------------------

**Highly Recommended books (You must read to master Deep learning, Tensor Flow, Data Science)**

- Python Data Science Handbook: Essential Tools For Working With Data by Jake VanderPlas
- HANDS ON MACHINE LEARNING WITH SCIKIT LEARN, KERAS & TENSORFLOW 2 by Aurelien Geron
  (This book is in two parts the second part is more important it talks about Tensorflow basics-advance, Here you'll learn RNN, CNN, Neural Networks etc)

- Read a paper which is very important (Attention is all you need) 
 [Link to the Document](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)


-----------------------------------------------------------------------

**Requirements**

To run this project, ensure that your environment meets the following requirements:

	•	Python 3.6 or higher
	•	Torch 1.7.0 or higher
	•	TorchVision 0.8.1 or higher
	•	Captum 0.4.0 or higher
	•	Pandas 1.1.5 or higher
	•	Matplotlib 3.3.2 or higher
	•	Pillow 7.2.0 or higher
	•	Google Colab (for running the notebook)
	•	TensorBoard 2.3.0 or higher (for visualizing the training process)
 
-----------------------------------------------------------------------

-----------------------------------------------------------------------
I was not able to upload the data set as the file was too large. If you want the exact set of data with which i am working.
Here is the link to -  [Data Set](https://www.kaggle.com/datasets/sumn2u/riped-and-unriped-tomato-dataset).


-----------------------------------------------------------------------
Git Clone the repository 
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

## Transfer Learning

There's multipel versions of pretrained models for ResNet. V1 has less accuracy (the oldest version) and V2 has the newest version (the new veresion).

There are several reasons why old versions of models and weights are maintained and made available even when newer versions with better performance exist. Here are some key reasons:

1. Backward Compatibility Existing Workflows: Many organizations and developers have existing workflows, scripts, and models that rely on older versions of the weights. Updating these workflows to use newer versions might require significant changes and testing. Reproducibility: Scientific research and publications often cite specific versions of models and weights. Keeping older versions ensures that results can be reproduced and validated by others.

2. Performance Trade-offs Inference Speed: In some cases, newer versions of weights might provide better accuracy but at the cost of increased computational resources or longer inference times. Users might prefer older versions for applications where speed is more critical than accuracy. Memory Usage: Newer models might require more memory, making them unsuitable for deployment on devices with limited resources.

3. Baseline Comparisons Benchmarking: Older versions serve as baselines for comparing the performance of new models and weights. This is crucial for understanding the improvements and trade-offs of newer versions. Algorithm Development: Researchers and developers often need to compare their new algorithms against established baselines to demonstrate improvements.

4. Model Training and Fine-Tuning Transfer Learning: Some users may prefer to start with older weights for specific transfer learning tasks, depending on the characteristics of their datasets or the specific features learned by the older weights. Training Stability: Older weights might be preferred in certain scenarios where they have shown to provide more stable training or convergence properties for specific tasks.

5. Historical Context Legacy Systems: Some legacy systems and applications are built with older versions of models. Changing these systems might not be feasible due to regulatory, technical, or financial constraints. Documentation and Tutorials: Many educational resources, tutorials, and documentation are built around older versions of models. Maintaining these versions ensures that learners and practitioners can follow along with existing educational material.
   
-----------------------------------------------------------------------

## what is quantized machine learning/ quantized weights

Quantization has couple benefits and concepts:

1. Floating point to integer:
-quantization typically involves converting 32-bit floating point numbers (FP32) to lower precision formats such as 8 bits (INT8).

2. Efficiency improvement: Memory Footprint: Lower precision numbers require less memory, leading to a reduced memory footprint for the model. Inference Speed: Integer arithmetic operations are faster and more power-efficient than floating-point operations, resulting in faster inference times and lower power consumption.
   
3. Types of quantization:
- post: the model is trained in full precision and quantization is applied after training. Small loss of accuracy but simpler.

- pre(quantization aware training): Model is trained with quantization in mind, simulating the effects of quantization during the training process. Preserves more accuracy.

- se Cases:
Good for mobile devices/applications where computational power and battery life are constrained.

- Use Cases:
Regular Weights: Preferred for training and tasks requiring high precision and large computational resources. Quantized Weights: Preferred for deployment and inference on resource-constrained devices where speed and efficiency are prioritized over minimal accuracy loss.

Quantized Weights: Often require a process called Quantization Aware Training (QAT) or post-training quantization to convert the FP32 weights to INT8 while attempting to minimize the impact on model accuracy.

Example in Context For instance, in the context of the MobileNetV3

model: MobileNet_V3_Large_QuantizedWeights.

IMAGENET1K_QNNPACK_V1:

These quantized weights are optimized for inference on CPUs using QNNPACK backend, suitable for mobile and edge devices.

MobileNet_V3_Large_Weights.

IMAGENET1K_V2: These are regular FP32 weights, providing slightly better accuracy and suitable for environments where computational resources are less constrained.

Important Note about quantization
PyTorch supports INT8 quantization compared to regular FP32 models(float) for a 4x reducton in the model size and 4x reduction in memory bandwidth requirements.

-----------------------------------------------------------------------

## How quantization works:

Symmetric quantization:
The range of the floating-point numbers is symmetrically distributed around zero.

Scaling factor:
s = max(abs(min), abs(max)) / (2^b-1 - 1)

zero point = z = 0

- quantization: q = round(x/s)

- dequantization: x = q * s

- Asymmetric quantization: In asymmetric quantization, the range of the floating-point numbers is not necessarily centered around zero. This approach uses a zero point to handle cases where the distribution of values does not include zero or is not symmetric around zero.

Scaling factor: s = (max - min) / (2^b - 1)

Zero Point: z = round(-min/s)
quantization: q = round(x/s) + z

dequatization: x = (q-z)*s

''''''''''''''''''''''''''''''''''''general equation underneath:----------

The linear quantization:

q = round((x - min)÷s)

When we linearly dequantize:

x = q * s + min

s - Scaling value

This is the most important parameter.

s = (max-min)/(2^b - 1)

if you want 8 bit quantization, you put 8 in the b.

min = -0.8, max = 0.6

s = (0.6-(-0.8)) / 255 = 1.4/255 = 0.0055

Zero Point (z):

z = is the real number zero. z = for symmetric quantization, the zero point is usually zero. For asymmetric, it is z = -min/s

-----------------------------------------------------------------------

## Quantization-Aware Training (Pre quantization)

During training, quantization-aware training (QAT) simulates quantization effects in the forward and backward passes to improve the robustness of the model when weights and activations are quantized during inference.

Fake Quantization: In QAT, "fake" quantization is applied where values are quantized and dequantized during training:

quantized x = s * round(x/s)

This ensures that the model learns weights that are robust to quantization.

Gradient Propagation: During backpropagation, gradients are calculated based on the fake quantized values, allowing the model to adjust the weights to minimize the quantization error.

-----------------------------------------------------------------------

## Post Quantization

Post-training quantization (PTQ) involves training the model with full precision and then quantizing it afterward. This can be done in several ways:

Static Quantization: Calibrate the model using a representative dataset to determine the appropriate scale and zero points.

Dynamic Quantization: Quantize weights statically but dynamically quantize activations during inference.

-----------------------------------------------------------------------

## Pre-training Quantization 

Pre-training quantization is the process of training a neural network directly with quantized weights and activations from the beginning. This approach is also known as Quantization-Aware Training (QAT).

Post-training Quantization Post-training quantization is the process of converting a fully trained model (using full precision weights) to a quantized version after the training has completed. This is also known as Post-Training Quantization (PTQ).

Pre-training Quantization (QAT) involves training a model with quantization effects simulated during training, allowing the model to learn and adjust for quantization-induced errors, often resulting in higher accuracy for the quantized model. Post-training Quantization (PTQ) involves converting a fully trained model to a quantized version, offering simplicity and flexibility at the potential cost of a slight drop in accuracy, which can be mitigated using calibration techniques.

-----------------------------------------------------------------------

### GOAL 
My goal is to reach above 90% accuracy with my model.

### GOAL ACHIEVED
My model was able to achieve 83% accuracy.
