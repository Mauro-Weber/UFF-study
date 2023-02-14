#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:21:24 2023

@author: weber
"""
# Import SparkSession
from pyspark.sql import SparkSession

# Import Pandas and Numpy
import numpy as np
import pandas as pd

spark = SparkSession.builder.appName("SparkKnn").getOrCreate()


# make a list with points using coordinates x and y from a dataframe 
# tha contains columns x and y

def getListOfPoints(dataframe):
    list_points = []
    data_collect = dataframe.collect()
    
    # create list with distance from point
    for row in data_collect:
        x = float(row["x"])
        y = float(row["y"])
        point = (x,y)
        list_points.append(point)
    
    return(list_points)

# Create Values of x and y
xvalues = np.random.rand(10).tolist()
yvalues = np.random.rand(10).tolist()

# Create DataFrame With Coordinates
first_dataframe = spark.createDataFrame(zip(xvalues,yvalues), schema=['x', 'y'])

#print(getListOfPoints(first_dataframe))