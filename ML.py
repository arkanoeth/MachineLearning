#Import all of the necesary modules
from sklearn.datasets import load_wine #Wine datasetload it into a variable
from sklearn.preprocessing import StandardScaler #Scale and normalizer
from sklearn.model_selection import train_test_split #To splip the data into traning and validation
from sklearn.neighbors import KNeighborsClassifier #Import the KNN algorithm
from sklearn.metrics import accuracy_score #To evaluate accuracy 
from sklearn.tree import DecisionTreeClassifier #Decision tree algorithm
from sklearn.decomposition import PCA #PCA class
from sklearn.cluster import KMeans #Kmeans 
from sklearn.cluster import MiniBatchKMeans #MiniBatchKMeans
import matplotlib.pyplot as plt #For the scatter plot 
from string import Template
from IPython.display import display
from base64 import b64encode
from jinja2 import Template

wineData = load_wine()

#Split the data into features (X) and labels (y)
X = wineData.data
y = wineData.target

#Data scaling and normalization
scaler = StandardScaler()
XScaled = scaler.fit_transform(X)

#Training and data sets
XTrain, Xtest, YTrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42)
XScaledTrain, XScaledTest, YScaledTrain, YScaledTest = train_test_split(XScaled, y, test_size=0.2, random_state=42)

#End of the 1 item

#I choose 2 for the k value
k = 2

#Train the KNN model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(XTrain, YTrain)
knn_scaled = KNeighborsClassifier(n_neighbors=k)
knn_scaled.fit(XScaledTrain, YScaledTrain)

#Make predictions on the validation sets
yPred = knn.predict(Xtest)
YScaledPred = knn_scaled.predict(XScaledTest)

#Find the accuracy of the model
accuracyKNN = accuracy_score(Ytest, yPred)
accuracy_scaledKNN = accuracy_score(YScaledTest, YScaledPred)

#print("Accuracy on original data:", accuracyKNN)
#print("Accuracy on scaled data:", accuracy_scaledKNN)

#End of the 2 item 

#I choose 4 for max_depth
MD = 4

#Train the decision tree model
dt = DecisionTreeClassifier(max_depth=MD)
dt.fit(XTrain, YTrain)
dt_scaled = DecisionTreeClassifier(max_depth=MD)
dt_scaled.fit(XScaledTrain, YScaledTrain)

#Make predictions on the validation sets
yPred = dt.predict(Xtest)
YScaledPred = dt_scaled.predict(XScaledTest)

#Find the accuracy of the model
accuracyDT = accuracy_score(Ytest, yPred)
accuracy_scaledDT = accuracy_score(YScaledTest, YScaledPred)

#print("Accuracy on original data:", accuracyDT)
#print("Accuracy on scaled data:", accuracy_scaledDT)

#End of 3 item

#Fit the PCA model to the scaled dataset
pca = PCA(n_components=2)
pca.fit(XScaledTrain)

#Transform the scaled dataset into the 2-dimensional PCA space
XPca = pca.transform(XScaledTest)

#plot the scatter graph
plt.scatter(XPca[:, 0], XPca[:, 1], c=YScaledTest)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Scaled Data')

plt.savefig("Plot1.png")

#End of 4 item

#Create KMeans and MiniBatchKMeans objects
kmeans = KMeans(n_clusters=3, random_state=42)
mbkmeans = MiniBatchKMeans(n_clusters=3, random_state=42)

#Fit both models to the transformed dataset
kmeans.fit(XPca)
mbkmeans.fit(XPca)

#Predict the cluster assignments of each data point using both models
y_kmeans = kmeans.predict(XPca)
y_mbkmeans = mbkmeans.predict(XPca)

#Configure PLT for make the graph of kmeans and mbkmeans
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(XPca[:, 0], XPca[:, 1], c=y_kmeans)
ax1.set(title='KMeans', xlabel='PCA 1', ylabel='PCA 2')

ax2.scatter(XPca[:, 0], XPca[:, 1], c=y_mbkmeans)
ax2.set(title='MiniBatchKMeans', xlabel='PCA 1', ylabel='PCA 2')

#Plot all the graphs
plt.savefig("Plot2.png")

# Define the variables to be inserted into the template
title = 'Machine Learning With Python'
item1 = round(accuracyKNN,ndigits=3)
item2 = round(accuracy_scaledKNN,ndigits=3)
item3 = round(accuracyDT,ndigits=3)
item4 = round(accuracy_scaledDT,ndigits=3)


with open('Plot1.png', 'rb') as f:
    image1 = b64encode(f.read()).decode()
with open('Plot2.png', 'rb') as f:
    image2 = b64encode(f.read()).decode()



# Define the HTML template
HtmlTemplate = Template('''
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        h1 {
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <h2>KNN Accuracy</h2> 
    <p>
        <strong>K selected:</strong> {{ k }}
        <br>
        <br>
        <strong>Original Data:</strong>{{ item1 }}
        <br>
        <br>
        <strong>Scaled Data:</strong>{{ item2 }}
        <br>
        <h4>In this case we see that the results obtained by the model with the scaled and normalized data generate better estimates.</h4> 
        <h2>Decision Tree Accuracy</h2>
        <strong>Max Depth selected:</strong> {{ MD }}
        <br>
        <br>
        <strong>Original Data:</strong>{{ item3 }}
        <br>
        <br>
        <strong>Scaled Data:</strong>{{ item4 }}
        <br>
        <h4>The decision tree starting from any of the two data sets obtains the same accuracy.</h4>
    <img src="data:image/png;base64,{{ image1 }}" alt="Graph 1">
    <img src="data:image/png;base64,{{ image2 }}" alt="Graph 2">
    </p>
</body>
</html>
''')


# Render the template with the variables
HtmlOutput = HtmlTemplate.render(title=title, item1=item1, item2=item2, item3=item3, item4=item4, k=k, MD=MD, image1=image1, image2=image2)

# Write the output to a file
with open('my_page.html', 'w') as f:
    f.write(HtmlOutput)

import webbrowser
webbrowser.open('my_page.html')