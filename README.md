# Machine Learning Code

This Python script was made for demonstrates various machine learning techniques using the wine dataset.

## Dataset
The wine dataset is used in this script. It is included in the scikit-learn library, so you don't need to download it separately.

## Dependencies
Make sure you have the following dependencies installed before running the code:

- scikit-learn
- matplotlib
- string
- IPython
- base64
- jinja2

You can install these dependencies using the pip package manager by running the following command:

_pip install scikit-learn matplotlib IPython jinja2_

## Running the Code

To run the code, simply execute the Python script. Ensure that you have the necessary datasets and dependencies installed.

The script will perform various machine learning tasks such as data preprocessing, model training, evaluation, and visualization. The results will be saved in the form of accuracy scores and scatter plots.

Once you have installed the dependencies and executed the script, you should see the results of the machine learning tasks such as:

- Imports the necessary modules for data preprocessing, model training, evaluation, and visualization.
- Loads the wine dataset and splits it into features (X) and labels (y).
- Performs data scaling and normalization using the StandardScaler class.
- Splits the data into training and validation sets.
- Trains a K-Nearest Neighbors (KNN) model on both the original and scaled datasets.
- Makes predictions on the validation sets using the trained KNN models.
- Calculates the accuracy of the KNN models.
- Trains a Decision Tree model on both the original and scaled datasets.
- Makes predictions on the validation sets using the trained Decision Tree models.
- Calculates the accuracy of the Decision Tree models.
- Applies Principal Component Analysis (PCA) to the scaled dataset and transforms it into a 2-dimensional space.
- Plots a scatter graph of the transformed data.
- Saves the scatter graph as "Plot1.png".
- Creates KMeans and MiniBatchKMeans objects and fits them to the transformed dataset.
- Predicts the cluster assignments of each data point using both KMeans models.
- Plots the results of KMeans and MiniBatchKMeans clustering.
- Saves the clustering graphs as "Plot2.png".
- Defines the variables and HTML template for generating a report.
- Renders the template with the variables.
- Writes the output to an HTML file named "my_page.html".
- Opens the HTML file in a web browser.

## Output of the code:

![image](https://github.com/arkanoeth/MachineLearning/assets/62271657/4de5398a-e953-4f62-90e5-d16462fed963)


