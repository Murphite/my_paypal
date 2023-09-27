# Welcome to My Paypal
***

## Task
To build a fraud detection model using the provided dataset and help FriendPay, a competitor of PayPal, and increase revenue from transaction fees.

## Description
The following steps were followed:

1. Data Collection/Cleaning: Obtain the credit card transaction dataset and clean it by removing any missing values, duplicates, or outliers.

2. Data Exploration: Explore the data to gain insights into its structure, distribution, and relationships between the variables. This will help determine which features are most important in predicting fraud.

3. Data Visualization: Visualize the data to identify patterns, correlations, and anomalies that may be useful in detecting fraud. Use graphs, charts, and other visual tools to present your findings.

4. Machine Learning: Train and test various machine learning algorithms such as logistic regression, decision trees, and random forests on the data to develop a fraud detection model. Evaluate the model's performance using the AUPRC.

6. Communication: Prepare a presentation with slides on how the model works, including assumptions, implications, and other important information. Provide the DevOps team with code that can be pushed to production, a transaction data simulator, and a technical specification for integrating the model into FriendPay's existing system.

## Installation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import re
import string

!pip install scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix 

## Usage
* Loading and inspecting the dataset using pandas to understand the data structure and format.
* Checking for missing values, outliers, and imbalanced classes in the dataset.
* Visualizing the distribution of the features using histograms, boxplots, and density plots to identify any patterns or anomalies in the data.
* Examining the correlation between features using a correlation matrix and heatmap to identify any highly correlated features.
* Visualizing the class distribution using a bar chart to understand the balance between fraudulent and non-fraudulent transactions.
* Using scatter plots to visualize the relationship between two features and how they relate to the target variable.
* Using Principal Component Analysis (PCA) to visualize the dataset in a 2D and 3D space.
* Data preprocessing: Standardization of the data using StandardScaler() to make sure all features are on the same scale, and reducing the number of dimensions using principal component analysis (PCA) to two components.
* Data splitting: The data is split into training and testing sets using train_test_split().
* Model training: A logistic regression model is trained on the training data using LogisticRegression(), DecisionTreeClassifier(), and RandomForestClassifier().
* Model evaluation: The performance of the model is evaluated on the test data using evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix using the evaluate_classification_model() function.

```
./my_project argument1 argument2
```

### The Core Team
Murphy Ogbeide

<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt='Qwasar SV -- Software Engineering School's Logo' src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px'></span>
