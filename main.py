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
from euclidean_distance import euclidean_distance
from knn import kNearestNeighbor
from get_points import getListOfPoints


spark = SparkSession.builder.appName("SparkKnn").getOrCreate()


dataframe = spark.read.parquet("./dataframe.parquet")
second_dataframe = dataframe.withColumn("partition", spark_partition_id())


knn_dictionary = {}
knn_list = []
for partition in range(second_dataframe.rdd.getNumPartitions()):
    third_dataframe = second_dataframe.filter(second_dataframe.partition == partition)
    for point in getListOfPoints(third_dataframe):
        knn_list = kNearestNeighbor(third_dataframe, point, 2)
        knn_dictionary.update({point: [knn_list, partition]})
        
#print(knn_dictionary)


lista_key = []
lista_values1 = []
lista_values2 = []
for key in knn_dictionary:
    lista_key.append(str(key))
    lista_values1.append(str(knn_dictionary[key][0]))
    lista_values2.append(str(knn_dictionary[key][1]))
    
dict_dataframe = spark.createDataFrame(zip(lista_key,lista_values1,lista_values2), schema=['point', 'knn_points', 'partition'])
#print(knn_dictionary)
dict_dataframe.show()

# save rersult
dict_dataframe.write.csv("topresults.csv")

