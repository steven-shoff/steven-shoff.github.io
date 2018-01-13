---
layout: post
title: "LDA Classifier"
modified:
categories: ml
excerpt:
tags: [Machine Learning, Classifier, LDA]
images:
date: 2017-12-01T15:39:55-04:00
modified: 2017-12-02T14:19:19-04:00
---

## Method:       Linear Discriminant Analysis

Types:        Supervised Learning - Classifier

Requirement:  Training Instances with labels

LDA is a classifier with linear decision surface. It performed supervised learning by projecting input data into a linear subspace with direction which maximize distances between classes.

In this example, we will use 'digit' dataset in sklearn as training and test data. For giving simple tutorials to beginners in ML, we will pick digit '1' and '7' only.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_digits
from sklearn import cross_validation

# 1. Data Preparation

digits = load_digits()                          # Load digit dataset
data = digits['data']
images = digits["images"]
target = digits["target"]

num1, num2 = 1,7                              # Data filtering
mask = np.logical_or(target == num1, target == num2)
filtered_data = data[mask]
filtered_target = target[mask]

filtered_target[filtered_target == num1] = 0   # Relabel targets
filtered_target[filtered_target == num2] = 1

print "Filtered dataset has {} instances.".format(len(filtered_target))
```

In this example, we try to distinguish '1' and '7' from the dataset:
![digit1]({{ "/images/Classifier/digit1.png" }})
![digit7]({{ "/images/Classifier/digit7.png" }})

To perform dimension reduction, we start with hand-crafted method first. Advanced dimension reduction techniques (Linear Regression, Ridge Regression with Regularization...etc will be shown in other chapters. 

```python
# 2. Dimension Reduction

def reduce_dim(x):

    instance_num = np.shape(x)[0]
    reduced_x = np.zeros((instance_num, 2), dtype = "float64")
    
    imgn1 = filtered_data[filtered_target == 0, :]
    imgn2 = filtered_data[filtered_target == 1, :]

    imgn1 = np.mean(imgn1, axis = 0)
    imgn2 = np.mean(imgn2, axis = 0)

    img_diff = np.reshape(imgn1 - imgn2, (8, 8))
    
    # If you want to visualize the result, uncomment the following lines:
    # plt.figure()
    # plt.imshow(img_diff, interpolation = "nearest")   
    # plt.figure()
    # plt.imshow(images[num1]-images[num2], interpolation = "nearest")    
    # plt.show()
    
    feature_1 = 1*x[:,12] + 4*x[:,19] +1*x[:,44]+5*x[:,52]+5*x[:,60]
    feature_2 = 2*x[:,5] + 5*x[:,6] +3*x[:,14]+2*x[:,50]+3*x[:,58]
    
    reduced_x = np.asarray([feature_1, feature_2]).T
    return reduced_x
    
reduced_data = reduce_dim(filtered_data)    
print 'The shape of x before reduction = {}.\n'.format(np.shape(filtered_data))
print 'The shape of x after reduction = {}.\n'.format(np.shape(reduced_data))
```
Digit data after dimension reduction:

```python

# 3. Data Visualization
plt.figure()
reduced_n1 = reduced_data[filtered_target == 0]
reduced_n2 = reduced_data[filtered_target == 1]
plt.scatter(reduced_n1[:, 0], reduced_n1[:, 1], color = 'red', marker = 'x')
plt.scatter(reduced_n2[:, 0], reduced_n2[:, 1], color = 'green', marker = '.')
plt.show()
```

![data_dim]({{ "/images/Classifier/classifier_data.png" }})

Then we start training the classifier with training instances and labels

```python
# 4. Filtered dataset spliting

X_train, X_test, y_train, y_test = cross_validation.train_test_split(reduced_data, filtered_target, test_size = 0.4, random_state = 0)
# print np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)

# 5. LDA Training and Testing Function:

def fit_lda(training_features, train_labels):
    
    fea0 = training_features[train_labels == 0, :]
    fea1 = training_features[train_labels == 1, :]

    mu0 = np.mean(fea0, axis = 0)
    mu1 = np.mean(fea1, axis = 0)

    mu = np.array([mu0, mu1])
    
    covmat = np.cov(training_features.T)
    
    p1 = float(np.count_nonzero(train_labels))/len(train_labels)
    p = np.array([1-p1, p1])
    
    return mu, covmat, p

def predict_lda(mu, covmat, p, test_features):
    
    predicted_labels = np.zeros(len(test_features))
    
    for j in range(0,len(test_features)):
        
        qd_result = []
        
        for i in range(0,len(mu)):
            fun_val = (-1/2)*np.log(np.linalg.det(covmat)) + (-1/2)*np.dot(np.dot((test_features[j] - mu[i]),np.linalg.inv(covmat)),(test_features[j] - mu[i]).T) + np.log(p[i])
            qd_result.append(fun_val)
        
        predicted_labels[j] = np.argmax(np.array(qd_result), axis = 0)     
    return predicted_labels

# 6. LDA Implementation

# 6a. Training part:
mu_lda, covmat_lda, p_lda = fit_lda(X_train, y_train)  
training_estimate_lda = predict_lda(mu_lda, covmat_lda, p_lda, X_train)

# 6b. Testing part
test_estimate_lda = predict_lda(mu_lda, covmat_lda, p_lda, X_test)
```
To evaluate the accuracy/performance of the clssifier, we simply calculate the error rate of training set and test set

```python
# 7. Error Evaluation:
LDA_trainerr = np.sum(np.abs(training_estimate_lda - y_train)) / float(y_train.shape[0])
print "Error rate of  LDA Classifer with training data = {}".format(LDA_trainerr)

LDA_testerr = np.sum(np.abs(test_estimate_lda - y_test)) / float(y_test.shape[0])
print "Error rate of  LDA Classifer with test data = {}".format(LDA_testerr)
```

If you are interested in visualizing the decision region, you can also implement the following code:

```python

# Decision Region Visualization

cmap_light = ListedColormap(['lightcoral', 'lightblue'])
cmap_bold = ListedColormap(['coral', 'darkcyan'])

x_min, x_max = X_train[:, 0].min() - 10, X_train[:, 0].max() + 10
y_min, y_max = X_train[:, 1].min() - 10, X_train[:, 1].max() + 10

#Create 426x252 pixels image
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 1))

Z = predict_lda(mu_lda, covmat_lda, p_lda, np.c_[xx.ravel(), yy.ravel()])
Z = np.asarray(Z).reshape(xx.shape)

plt.figure()
splot = plt.subplot(111, aspect='equal')
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.contourf(xx, yy, Z > 0.5, alpha=0.5)
plt.scatter(X_train[:,0] ,X_train[:,1], c=y_train, 
            cmap=cmap_bold, edgecolor='k', s=20)
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')
plot_ellipse(splot, mu_lda[0], covmat_lda, 'r')
plot_ellipse(splot, mu_lda[1], covmat_lda, 'royalblue')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Visualization")

plt.show()
```
