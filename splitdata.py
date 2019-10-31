import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
"""columns = "index,sepal length,sepal width,petal length, petal width,class".split(",")""" 
# Declare the columns names
iris = pd.read_csv("iris.csv") # Call the diabetes dataset from sklearn
# load the dataset as a pandas data frame
#y = iris.target # define the target variable (dependent variable) as y
# create training and testing vars
X,Y= train_test_split(iris, test_size=0.2)
X.to_csv("train.csv")
Y.to_csv("test.csv")