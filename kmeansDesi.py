#!/usr/bin/env python
# coding: utf-8

# # Software Engineering Bootcamp - Level 3 Task 22

# ''' For the kMeans clustering problem for this task I followed the outline in the pdf and also used the shell template
# provided in this task. Essentially I followed 5 steps and then finally put it all together. 
# Part 0 - I imported all the required python libraries
# Part 1 - I defined a function to read in the csv file as a pandas dataframe
# Part 2 - I defined a function to calculate the euclidean distance between points
# Part 3 - I defined a function to randomly select centroids based on the number of cluster chosen by the user
# Part 4 - I wrote a function to assign each point to a cluster (based on the current centroids coordinates) and I also wrote
# a function to return the sum of the euclidean distance between the points and their chosen cluster or centroid
# Part 5 - I wrote teh Kmeans algotrith to update the centroids based on the points assigned to the cluster. the centroids 
# were updated to mean of the points closest to it.
# 
# Lastly I put all the functions together to determine the cluster that each point belongs to aswell plot the scatter graph.'''
# 
# 

# ''' Part 0: Import Libraries '''

# In[ ]:


# Numpy allows us to do the math to the data
import numpy as np

# Numpy allows us to do dataframe and matrix computations
import pandas as pd
 
# Seaborn allows us to plot the graphs when it is time to plot nearer to the end of this task
import seaborn as sns

# And Finally random, because we use random to find and initialize the clusters that we need to calculate the distance to
import random 


# '''
# Part I : We know that there are 3 datasets we need to be able to read when the user decides. So first we should create
#         a function that allows us to read any of the three csv files (Keep in mind that Birthrate is x and Life Expectancy
#                                                                       is y)
# '''

# In[ ]:


# Define function with one argument since we will be using a csv file to be able to... Well, y'know. Read the file.
# The read file should also be returned as a list for use within our calculations. This can also be done outside this function

#For this function I imported the data as a pandas dataframe

def readCSV(file):
    
    df = pd.read_csv(file, delimiter=',')
    
    return df
    


# '''
# We are now able to read whichever file the user chooses. Now we get into the mathematical part of things. (No worries,
#     numpy makes this easy for us)
# 
# Part II : Finding the Euclidean distance between any two points. This is one of the math parts of the task
#             The PDF contains what the Eucliedian distance calculation it is. It's up to you for how the calculation is done
#                 in Python, HINT : np.sqrt allows you to calculate the square root of a value.
#                     Also keep in mind that you can use nested loops to separate x1 and x2 and y1 and y2
# '''

# In[ ]:


#For this function I calculated the distance using the lambda function and applying it the pandas dataframe. The pandas
# is essentially the data from the csv file that was read in. The read function above imports the data as a dataframe.

def euclideanDistCalc(arr, x1, y1):  
        
    arr.columns = ['countries', 'birthRate', 'lifeExpectancy']
    arr = arr.assign(sumProduct=lambda x: (((x['birthRate'] - x1)**2 + (x['lifeExpectancy'] - y1)**2)**0.5))

    result = arr['sumProduct'].values
    
    del arr['sumProduct']

    return result


# '''
# Part III : Initalizing the Clusters. This part is determined by the amount of clusters the user wants in their analysis. 
#         This can be done easily by creating two empty lists and appending (using a for loop) the data from the actual dataset (we can refer to this in 
# 	our function as strList) then we append this data to our empty X and Y lists and use .random to take a random sample for our cluster initialization.
# '''

# In[ ]:


#The below function randomly picks points in our data as our initial centroids. The number of centroids chosen is equal
#to the number of clusters selected by the user.

def initCentroids(arr, numClusters):
    
    random.seed(42)
    
    arr.columns = ['countries', 'birthRate', 'lifeExpectancy']
    rows = len(arr)
   
    x = []
    y = []
    
    for i in range(0,numClusters):
        j = random.randint(0,rows)
        x.append(arr['birthRate'][j])
        y.append(arr['lifeExpectancy'][j])
        
    return x, y


# '''
# Part IV : The below two functions essentially returns the cluster that the respective point belongs to (returnCluster). In
# addition I've also defined a function to calculate the sum of the euclidean distance for all the points (returnDist). I will
# use the second function to check for convergence.
# '''

# In[ ]:


#For a given dataframe and a set of centroids, the below function calculates the cluster the point belongs to.

def returnCluster(arr, a, b):
    arr.columns = ['countries', 'birthRate', 'lifeExpectancy']
    count = len(a)
    rows = len(df)
    
    temp = pd.DataFrame()
    
    for i in range(0, count):
        temp['dist'+str(i)] = euclideanDistCalc(arr, a[i], b[i])

    cluster = []
    
    for k in range(0, rows):
        group = np.argmin(temp.iloc[k].values)
        cluster.append(group)
    
    return cluster


#The below function determines the sum of the euclidean distances for each point in the dataset

def returnDist(arr, a, b):
    arr.columns = ['countries', 'birthRate', 'lifeExpectancy']
    count = len(a)
    rows = len(df)
    
    temp = pd.DataFrame()
    
    for i in range(0, count):
        temp['dist'+str(i)] = euclideanDistCalc(arr, a[i], b[i])

    dist = 0
    
    for k in range(0, rows):
        group = np.min(temp.iloc[k].values)
        dist += group
    
    return dist


# '''
# Part V : And finally for our last bit, we'll be implementing the kMeans algorithm. This can be really easy (with proper research) or you can be 
# 	extra and create a function yourself! 
# 
# '''

# In[ ]:


#Essentially what this function does is determine the cluster for each point and based on this it then determines the mean
#for each points belonging to a specific centroid. The centroids are then updated with the mean values.


def kMeansAlg(arr, a, b):
    #    # If you're feeling tough
    count = len(a)

    cluster = returnCluster(arr, a, b)

    arr['cluster'] = cluster
        
    for j in range(0, count):
        a[j] = (arr[(arr['cluster'] == j)])['birthRate'].mean()
        b[j] = (arr[(arr['cluster'] == j)])['lifeExpectancy'].mean()
        

    del arr['cluster']
    
   
    return a, b


# '''
# Finally :We put all the functions defined above together to come up with our k-Means Clustering Algorithm.
# 
# '''

# In[ ]:


#Ask user for the file they want to import
dataSet1 = input('''Plese enter the file name you want to use: data1953.csv, data2008.csv or dataBoth.csv: ''' )    
                                                                                        
# Asks user for the number of clusters the want to split the data into
cluster_amount = int(input("Input cluster amount: ")) 

#Here we read in the file as a pandas dataframe and rename the columns so that its easier to use.
df = readCSV(dataSet1)
df.columns = ['countries', 'birthRate', 'lifeExpectancy']

#Init Centroids
a, b = initCentroids(df, cluster_amount)

#Init sum of the euclidean distance between each point in the dataset
dist = returnDist(df, a, b)

#The below variable will represent our iteration counter
maxIterations = 0

#Now we perform iterations to determine the optiomal centroids
#Essentially here update the centroids until the sum of the euclidean distance between the points does not improve or 
#the max number of interations doesn't exceed 20. The last part is there to ensure that we don't have an infinite loop.
while maxIterations <= 20:
    a,b = kMeansAlg(df, a , b)
    if (dist == returnDist(df, a, b)):
        break
    else:
        dist = returnDist(df, a, b)
        maxIterations += 1    


#Now based on our optimized centroids, we determine the cluster of each point in our dataset.
clust = returnCluster(df, a, b)

#Here I created a dictionary that will store the values/coordinates of the optmized centroids
dict = {'X': a, 'Y': b}

#Create a temporay dataframe that has the centroid coordinates
temp = pd.DataFrame(data=dict)

#the df['cluster'] column basically shows which cluster the point belongs to
df['cluster'] = clust


#Setup colour list for our clusters
cmaps = ["Blue", "Green", "Purple", "Pink", "Red", "Orange", "Brown", "Black", "Yellow", "Grey"] 
colormap=cmaps[0:cluster_amount]

markerColor = []
color = 'Red'
for i in range(0,len(a)):
    markerColor.append(color)

#Lastly we plot our data and colour each according to the cluster it belongs to
#I also ploy the centroid
print("\n\nCategorised ScatterPlot\n")

sns.scatterplot('birthRate', 'lifeExpectancy', data=df, hue='cluster', palette=colormap)
sns.scatterplot('X', 'Y', data=temp, hue=temp.index, s=100, palette=markerColor, marker='x', legend= False)

#Finally delete the cluster columns and temp dataframe
del df['cluster']
del temp


# In[ ]:




