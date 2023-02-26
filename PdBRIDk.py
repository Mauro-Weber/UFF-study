#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:45:21 2023

@author: weber
"""

# Import SparkSession
from pyspark.sql import SparkSession

# Import Pandas and Numpy
from numpy.random import rand
from numpy.linalg import norm
from numpy import asarray, sum, argmin
from pyspark.sql.functions import udf

# import lit function
from pyspark.sql.functions import lit, monotonically_increasing_id, row_number, spark_partition_id
from pyspark.sql import functions as F
from pyspark.sql import Window


from pyspark.sql.types import StructType,StructField, StringType, FloatType

# import math 
import math



#functions session

# create function to get list of pivots

def get_pivots(n): 
    pivots_list = []
    for i in range(n):
        x = rand(1)
        y = rand(1)
        point = (float(x),float(y))
        pivots_list.append(point)
    return pivots_list


# create function to get list of points from a dataframe

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


# create function to find the closest pivot from a point

def closest_pivot(point, pivots_list):
    pivots_list = asarray(pivots_list)
    point = asarray(point)
    distances = norm(pivots_list-point, axis=1)
    min_index = argmin(distances)
    return tuple(pivots_list[min_index])


# create function to find the closest pivot from a point

def closest_point(pivot, points_list):
    points_list = asarray(points_list)
    pivot = asarray(pivot)
    distances = norm(points_list-pivot, axis=1)
    min_index = argmin(distances)
    return tuple(points_list[min_index])


# create function to get k Nearest Neighbor from a point 

def kNearestNeighbor(dataframe, point, k):
    
    results = []
    data_collect = dataframe.collect()
    
    # create column with distance from a given point 
    for row in data_collect:
        x = float(row["x"])
        y = float(row["y"])
        partition_point = (x,y)
        result = math.dist(point,partition_point)
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


# create function to implement BRIDk algorithm

def brid(dataframe, k, pivot):
    points_list = getListOfPoints(dataframe)

    result_list = []
    length_list = 0
    for i in range(len(points_list)):
        
        if i == 0:
            # get closest to pivot
            close_point = closest_point(pivot, points_list)
            
            # add closest point to list
            result_list.append(close_point)
            length_list += 1
            
            # get distance between pivot e closest point
            distance = math.dist(pivot,close_point)
            
            # save influence radius and last point 
            influence = distance
            influence_point = close_point
            
            points_list.remove(close_point)
        
        if i > 0:
            
            # get closest to pivot
            close_point = closest_point(pivot, points_list)
            
            # get distance between influence point and new closest point
            points_distance = math.dist(influence_point,close_point)
            
            # verify if point is in influence radius
            if points_distance <= influence:

                points_list.remove(close_point)
            
            if points_distance > influence:
                
                # get distance between pivot e closest point
                distance = math.dist(pivot,close_point)
                
                # save influence radius and last point 
                influence = distance
                influence_point = close_point
                
                
                result_list.append(close_point)
                length_list += 1
                points_list.remove(close_point)

        if length_list == k:
            break

    return result_list
    # reset lists
    result_list = []
    length_list = 0

#---------------------end-functions---------------------------


spark = SparkSession.builder.appName("SparkRandomCoordinatesDataset").getOrCreate()


# importing dataframe chapado
dataframe = spark.read.parquet("/home/weber/Documents/Database/knn-project/DataFrames/coordinates_hundred.parquet")


# chose random pivots 

chose_pivots_list = []
pivots_list = get_pivots(5)
for i in dataframe.collect():
        point = tuple((i["x"],i["y"]))
        pivot = closest_pivot(point,pivots_list)
        chose_pivots_list.append(pivot)
        

# add pivots to dataframe new column
# create index to combine dataframe with results from list of pivots

dataframe_pivots = spark.createDataFrame([(str(l),) for l in chose_pivots_list], ['pivots'])
dataframe_with_index = dataframe.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
dataframe_pivots_withindex = dataframe_pivots.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    
# join the two dataframes (poitns, distance from point)
dataframe_join = dataframe_with_index.join(dataframe_pivots_withindex, dataframe_with_index.row_idx == dataframe_pivots_withindex.row_idx)
dataframe_new = dataframe_join.sort(F.col("pivots")).drop("row_idx")

# partitionate dataframe by pivots

dataframe_new.write.option("header",True) \
        .partitionBy("pivots") \
        .mode("overwrite") \
        .parquet("/home/weber/Documents/Database/knn-project/DataFrames/dataframe_partition.parquet")


# read partitionated dataframe

dataframe_1 = spark.read.parquet("/home/weber/Documents/Database/knn-project/DataFrames/dataframe_partition.parquet")


# find the k points using BRIDk algorithm for each partition

final_points = []
for partition in range(dataframe_1.rdd.getNumPartitions()):
    new_dataframe = dataframe_1.filter(dataframe_1.pivots == str(pivots_list[partition]))
    
    # get pivot
    pivot = new_dataframe.head()['pivots']
    pivot = pivot[1:]
    pivot = pivot[:-1]
    pivot = tuple(map(float, pivot.split(', ')))
    
    bridk = brid(new_dataframe, 5, pivot)
    for point in bridk:
        final_points.append(point)


# create new dataframe with BRIDk result 

pts = []
for i in final_points:
    x = i[0]
    y = i[1]
    xy = (float(x),float(y))
    pts.append(xy)
    
deptSchema = ["x", "y"]
dataframe_2 = spark.createDataFrame(data=pts, schema= deptSchema)



# run BRIDk considering every point as a query object

brid_dictionary = {}
brid_list = []
for point in getListOfPoints(dataframe_2):
    brid_list = brid(dataframe_2, 5, point)
    brid_dictionary.update({point: [brid_list]})
    

# saving the result as a dataframe with columns 'point' and 'knn points'

lista_key = []
lista_value = []

for key in brid_dictionary:
    lista_key.append(str(key))
    lista_value.append(str(brid_dictionary[key][0]))
    
brid_dataframe = spark.createDataFrame(zip(lista_key,lista_value), schema=['point', 'brid_points'])

brid_dataframe.write.csv("bridkresults.csv")


# find the top k considering every point as a query object

knn_dictionary = {}
knn_list = []
for point in getListOfPoints(dataframe_2):
    knn_list = kNearestNeighbor(dataframe_2, point, 5)
    knn_dictionary.update({point: [knn_list]})
    

# saving the result as a dataframe with columns 'point' and 'knn points'

lista_key2 = []
lista_value2 = []

for key in knn_dictionary:
    lista_key2.append(str(key))
    lista_value2.append(str(knn_dictionary[key][0]))
    
dict_dataframe = spark.createDataFrame(zip(lista_key2,lista_value2), schema=['point', 'knn_points'])

dict_dataframe.write.csv("knnresults.csv")



