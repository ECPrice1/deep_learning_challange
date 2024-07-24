# Deep Learning Challange

## Overview of the Analysis

## Introduction

A nonprofit foundation named "Alphabet Soup" is looking to machine learning as a means to create a model that will help them identify which organizations they can donate to with the stated goal of achiveing a 75% success rate, or in machine learning terms, 75% accuracy. In this exercise I've created 4 different machine learning models to attempt to achieve that goal. 

#### What variable(s) are the target(s) for your model?
* Information on all past organization Alphabet Soup invested in was provided via a web link csv and converted into a pandas datafraem. The 'IS_SUCCESSFUL' column from application_df is the target variable, this is a binary yes or no showing whether or not the project was successful.

* #### What variable(s) are the features for your model?
* The feature variables used are:
  1. AFFILIATION — Affiliated sector of industry
  2. CLASSIFICATION — Government organization classification
  3. USE_CASE — Use case for funding
  4. ORGANIZATION — Organization type
  5. STATUS — Active status
  6. INCOME_AMT — Income classification
  7. SPECIAL_CONSIDERATIONS — Special considerations for application
  8. ASK_AMT — Funding amount requested
 
#### What variable(s) should be removed from the input data because they are neither targets nor features?
* Identification columns: The "EIN" and "NAME" columns are identification columns that typically provide unique identifiers for each organization. These columns usually have no direct impact on the target variable and can be dropped without affecting the model's accuracy.

### Compiling, Training, and Evaluating the Model

#### How many neurons, layers, and activation functions did you select for your neural network model, and why?
#### Attempt #1 (Accuracy # fill in)

* In the first neural network model, two-layer hidden layers were used with 80 and 30 neurons used in the first and second layer, respectively. There were 43 input values and this seemed like a good starting point with a sufficient number of neurons to handle the amount of inputs. The first layer utilizing a sigmoid activation function and the second using a relu activation. The output layer had a single neuron and sigmoid activation function to map the output to a range between 0 and 1, representing either a successful or unsuccessful outcome. This attempt was run with 200 epochs, the choice for that being somewhat random, but seeming that it may be sufficiently high to handle the level of complexity, but also possibly low enought to avoid overfitting.
* 
*** image ***

## Results

This resulted in a 72.99% accuracy rating. Notably, the accuracy showing on each epoch as the program ran through the test dataset seemed to be higher that the final accuracy rating from the test dataset, which I thought may suggest overfitting to the training data.

#### Attempt #2 (Accuracy # fill in)

*** Image ***

* Given my suspicion of potential overfitting, I kept the 2nd attempt almost exactly the same in strcuture with the same number of layers, neurons and utilized the same activation functions. Controlling those variables and keeping them the same would allow me to isolate the number of epochs and see if that would result in higher accuracy. Since it's relatively easy to just change that value and run the model again, I tested both higher and lower values, 1000, 100, 250 and finally 150 epochs (shown in the Jupyter Notebook, resulting in 72.77% accuracy), all of these values, both higher and lower seemed to result in a drop in accuracy and lead me to look at adjusting other parameters of the model.

#### Attempt #3 (Accuracy # fill in)

*** Image ***

In the third attempt, I utilized 4 hidden layers with the number of nodes and activation functions as follows: 
1 - 80 neurons, sigmoid activation
2 - 50 neurons, relu activation
3 - 30 neurons, tanh activation
4 - 30 neurons, relu activation

Since there didn't appear to be an increase in accuracy when changing the number of epochs up or down, I chose the inital epoch value from attempt 1 of 200. Initially, the change to the model was to keep the layer architecture the same for the first and final hidden layer, 80 neurons, sigmoid for the first layer and 30 neurons, relu for the final hidden layer, with one additional 50 neuron relu layer. This resulted in 72.73% accuracy. i was curious to see if adding an additional layer with a different activation function would make a diffence, so I added an additional layer (layer 3 shown in the notebook) with 30 neurons and a tanh activation function. The final version of this attempt did result in a 72.91% accuracy, and increase moving from 3 to 4 hidden layers. Ultimately, the accuracy was still lower than attempt #1 at nearly 73%. 

#### Attempt #4 (Accuracy # fill in)

*** Image ***


 




 
Neurons

* By increasing the number of neurons in a layer, the model becomes more expressive and can capture complex patterns in the data. This allows for better representation of the underlying relationships between the features and the target variable, potentially leading to higher accuracy.

Epochs

* Increasing the number of epochs gives the model more opportunities to learn from the data and adjust the weights. It allows the model to refine its predictions and find better parameter values, which can lead to improved accuracy. However, it's important to find a balance as increasing epochs excessively can lead to overfitting.

Layers

* Adding more layers can provide the model with additional capacity to capture and represent intricate relationships within the data. Each layer can learn different levels of abstraction, enabling the model to extract more meaningful features and potentially improving accuracy. Deep models with multiple layers have the ability to learn hierarchical representations of the data, which can be advantageous for complex problems.
  
Activation functions

* Introducing a different activation function, such as tanh, can affect how the model interprets and transforms the inputs. Different activation functions have different properties and can capture different types of non-linearities. By using tanh, it introduce a different non-linearity that may better suit the problem at hand, potentially leading to increased accuracy.





4. Utilizing an Automated Optimiser (such as a hyperparameter tuner):
   

* Automated optimisers, like hyperparameter tuners, systematically explore various combinations of hyperparameters, such as activation functions, number of layers, number of neurons, and epochs. This exploration can help identify the most optimal combination of hyperparameters for your specific problem, potentially leading to higher accuracy. It saves you from manually trying out different combinations and allows the optimiser to leverage its search algorithms to find the best configuration.
  



## Conclusion
The deep learning model that I have developed was unable to achieve accuracy higher than 73%. To further improve the model's performance, I can consider the following steps:

1. Adding more data:
   * Increasing the size of the training dataset can help the model learn from a larger and more diverse set of examples. This can improve the generalisation capability of the model and potentially lead to higher accuracy. Collecting additional data relevant to the classification problem could provide the model with more information to make better predictions.
  
2. Checking data cleaning:
   * Ensuring that the data is properly cleaned is crucial for model performance. Cleaning includes handling missing values, handling outliers, normalizing or standardizing features, and addressing any data quality issues. By thoroughly reviewing and cleaning the data, I can mitigate the impact of noise or irrelevant information that might be affecting the model's accuracy.
  
3. Exploring alternative machine learning algorithms:
   * Trying a different algorithm, such as Random Forest, can provide valuable insights into the importance of different features. Random Forest can measure feature importance based on how effectively each feature contributes to the overall prediction. This analysis can help identify the key predictor columns, allowing you to focus on the most informative features and potentially improve accuracy.

4. Identifying feature importance and selecting relevant attributes:
   * Analysing feature importance helps determine which attributes have the most significant impact on the output. By identifying and selecting the most important attributes, you can reduce the noise and complexity in the model. Focusing on the most relevant features can enhance the model's ability to capture meaningful patterns and improve accuracy.
  
5. Addressing high bias and outliers:
   * High bias in the model can be caused by outliers or skewed data points that deviate significantly from the majority of the dataset. Identifying and addressing these outliers can help improve the model's performance. Techniques such as outlier detection, data transformation, or stratified sampling can be applied to mitigate the impact of outliers and reduce bias in the model.

6. Binning the data:
   * Binning continuous variables can be useful in certain scenarios. It can help simplify the relationship between variables and the target variable by grouping similar values into bins. This can reduce the complexity of the model and make it more robust to noise or fluctuations in the data, potentially leading to improved accuracy.

In summary, to improve the deep learning model's performance, I would consider adding more data, ensuring proper data cleaning, exploring alternative algorithms, identifying feature importance, addressing bias and outliers, and applying data binning techniques. Each step aims to enhance the model's ability to capture relevant patterns and reduce noise, ultimately improving accuracy in the classification problem. It is important to iterate and experiment with these steps, evaluating the impact on model performance and fine-tuning as necessary.

# Project Outline

## Instructions

### Step 1: Preprocess the Data

1. Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.
2. Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
3. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
4. Drop the EIN and NAME columns.
5. Determine the number of unique values for each column.
6. For columns that have more than 10 unique values, determine the number of data points for each unique value.
7. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
8. Use pd.get_dummies() to encode categorical variables.
9. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
10. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every five epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

1. Use any or all of the following methods to optimize your model:
    * Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    * Dropping more or fewer columns.
    * Creating more bins for rare occurrences in columns.
    * Increasing or decreasing the number of values for each bin.
    * Add more neurons to a hidden layer.
    * Add more hidden layers.
    * Use different activation functions for the hidden layers.
    * Add or reduce the number of epochs to the training regimen.
2. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
3. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
4. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
5. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
6. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

