#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:36:09 2023

@author: weber
"""
# Import SparkSession
from pyspark.sql import SparkSession

# Import Pandas and Numpy
import numpy as np
import pandas as pd

# import lit function
from pyspark.sql.functions import lit, monotonically_increasing_id, row_number, spark_partition_id
from pyspark.sql import functions as F
from pyspark.sql import Window

from euclidean_distance import euclidean_distance
from get_points import getListOfPoints

spark = SparkSession.builder.appName("SparkKnn").getOrCreate()

def kNearestNeighbor(dataframe, point, k):
    
    results = []
    data_collect = dataframe.collect()
    
    # create column with distance from a given point 
    for row in data_collect:
        x = float(row["x"])
        y = float(row["y"])
        partition_point = (x,y)
        result = euclidean_distance(point,partition_point)
        results.append(result)
    
    fourth_dataframe = spark.createDataFrame([(l,) for l in results], ['distancefrompoint'])
    dataframe_withindex = dataframe.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    fourth_dataframe_withindex = fourth_dataframe.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    
    # join the two dataframes (poitns, distance from point)
    final_dataframe = dataframe_withindex.join(fourth_dataframe_withindex, dataframe_withindex.row_idx == fourth_dataframe_withindex.row_idx)
    final_df = final_dataframe.sort(F.col("distancefrompoint").asc()).drop("row_idx")
    
    
    # list the k nearest points
    newfinal_df = final_df.limit(k)
    #newfinal_df.show()
    final_lista = getListOfPoints(newfinal_df)
    
    #resultdataframe.show()
    return(final_lista)