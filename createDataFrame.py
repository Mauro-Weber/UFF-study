#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:07:57 2023

@author: weber
"""

# Import SparkSession
from pyspark.sql import SparkSession

# Import Pandas and Numpy
import numpy as np
import pandas as pd

# import functions
from pyspark.sql.functions import lit, monotonically_increasing_id, row_number, spark_partition_id
from pyspark.sql import functions as F
from pyspark.sql import Window

# import math 
import math

# import self made functions 
#from euclidean_distance import euclidean_distance
#from knn import kNearestNeighbor
#from get_points import getListOfPoints


spark = SparkSession.builder.appName("SparkKnn").getOrCreate()

# Create Values of x and y
xvalues = np.random.rand(1000).tolist()
yvalues = np.random.rand(1000).tolist()

# Create DataFrame With Coordinates
dataframe = spark.createDataFrame(zip(xvalues,yvalues), schema=['x', 'y'])

# partionate dataframe and add column with partition#
#new_dataframe = dataframe.repartition(5).withColumn("partition", spark_partition_id())
new_dataframe = dataframe.repartition(10)
new_dataframe.write.parquet("./dataframe.parquet")