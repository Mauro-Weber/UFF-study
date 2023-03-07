#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:42:53 2023

@author: weber
"""

# Import SparkSession
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, FloatType, StructType, StructField, IntegerType
from pyspark.sql.functions import spark_partition_id, col

## the distance function
from scipy.spatial import distance
## creates the feature vector
from pyspark.ml.feature import VectorAssembler
## import numpy
import numpy as np

from pyspark.sql.functions import max, abs

## start session
spark = SparkSession.builder.appName("SparkLAESAKnn").getOrCreate()




##DEFINES A UDF FROM L2DIST THE DATAFRAME FROM A QUERY SET
distance_udf = F.udf(lambda x,y: float(distance.euclidean(x, y)), DoubleType())

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())



##
##  LAESA-KNN
##

def knn_laesa(dataframe, oq, k):
    
    for i in range(dataframe.rdd.getNumPartitions()):

        if i == 0:
            ## PART_ONE
            ## take the first k rows and calculate the distance from Oq
            
            df_part0 = dataframe.where(spark_partition_id() == i).localCheckpoint()
            df_part0_new = df_part0.withColumn("dist_Oq", simpleF(oq)(F.col('fv')))
            df_part0_new = df_part0_new.orderBy("dist_Oq").limit(k)
            max_value2 = df_part0_new.agg(max('dist_Oq')).collect()[0][0]

        else:
            ## PART_TWO
            ## from the second partition forward 
            ## uses the lower bound to cut useless rows
    
            df_parti = dataframe.filter(spark_partition_id() == i).localCheckpoint()
            df_parti = df_parti.where(df_parti["|d(oq,p1)-d(oi,p1)|"] <= max_value2)
            df_parti_new = df_parti.withColumn("dist_Oq", simpleF(oq)(F.col('fv')))



            ## PART_THREE 
            ## append the remain rows into the last dataframe and take the new top-K
            
            df_part0_new = df_part0_new.union(df_parti_new).orderBy("dist_Oq").limit(k).localCheckpoint()
            max_value2 = df_part0_new.agg(max('dist_Oq')).collect()[0][0]
    
    resultSet1 = df_part0_new.select("id", "dist_Oq").orderBy('dist_Oq').limit(k)
    return(resultSet1)


##
## TESTING
## 

## 
## DEFINE OQ, K E PIVOT
##

## defines query object and 
oq = [np.random.rand(1)[0],np.random.rand(1)[0]]

k = 10

# defines pivot
pivot = [np.random.rand(1)[0],np.random.rand(1)[0]]

# distance from pivot to Oq
dist_pivot_oq = distance.euclidean(pivot,oq) 


##
## DATAFRAME CONFIG
##

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("coord_x", FloatType(), True),
    StructField("coord_y", FloatType(), True)])

#df = spark.createDataFrame(data = data, schema = schema)
df = spark.read.csv("/home/weber/Documents/coordDF3.csv",schema=schema)

##SMALL SANITY CHECK - @PRODUCTION TESTAR LOADING
df = df.na.drop()


## define the struct for the dimensional feature vector
cNames = df.columns
cNames.remove("id")
assembler = VectorAssembler(
    inputCols=cNames,
    outputCol="fv")

## appends the fv into the dataframe as column
df = assembler.transform(df)
df_new = df.withColumn(f'distances_oi_pivot', simpleF(pivot)(F.col('fv')))
df_with_dist = df_new.withColumn("|d(oq,p1)-d(oi,p1)|", \
    abs (dist_pivot_oq - \
    F.col('distances_oi_pivot')))

df_with_dist = df_with_dist.orderBy("|d(oq,p1)-d(oi,p1)|")

df_2 = df_with_dist.repartitionByRange(20, col('|d(oq,p1)-d(oi,p1)|'))


##
## CALLING LAESA_KNN FUNCTION
##

result = knn_laesa(df_2, oq, k)
result.show()


spark.stop