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
            df = dataframe.where(spark_partition_id() == i).localCheckpoint()
            df = df.withColumn("dist_Oq", simpleF(oq)(F.col('fv')))
            df = df.orderBy("dist_Oq").limit(k)
            max_value2 = df.agg(max('dist_Oq')).collect()[0][0]

        else:
            ## PART_TWO

            df_aux = dataframe.filter(spark_partition_id() == i).localCheckpoint()
            df_aux = df_aux.where(df_aux["|d(oq,p1)-d(oi,p1)|"] <= max_value2)
            df_aux = df_aux.withColumn("dist_Oq", simpleF(oq)(F.col('fv')))



            ## PART_THREE
            df = df.union(df_aux).orderBy("dist_Oq").limit(k).localCheckpoint()
            max_value2 = df.agg(max('dist_Oq')).collect()[0][0]
            
    resultSet1 = df.select("id", "dist_Oq").orderBy('dist_Oq').limit(k)
    
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
df = df.withColumn(f'distances_oi_pivot', simpleF(pivot)(F.col('fv')))
df = df.withColumn("|d(oq,p1)-d(oi,p1)|", \
    abs (dist_pivot_oq - \
    F.col('distances_oi_pivot')))

df = df.orderBy("|d(oq,p1)-d(oi,p1)|")

##
## CALLING LAESA_KNN FUNCTION
##

result = knn_laesa(df, oq, k)
result.show()


spark.stop