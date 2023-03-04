# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##
## IMPORT SESSION
##


# Import SparkSession
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, FloatType, StructType, StructField, IntegerType


## import numpy
import numpy as np
## the distance function
from scipy.spatial import distance
## creates the feature vector
from pyspark.ml.feature import VectorAssembler


##
## BUILDING UP
##

## start session
spark = SparkSession.builder.appName("SparkRandomCoordinatesDataset").getOrCreate()

##defines a udf from l2dist the dataframe from a query point
distance_udf = F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())


def knn(dataframe, oq, k):       
    
    ## define the struct for the dimensional feature vector
    cNames = dataframe.columns
    cNames.remove("id")
    assembler = VectorAssembler(
        inputCols=cNames,
        outputCol="fv")
    
    ## appends the fv into the dataframe as column
    df = assembler.transform(dataframe)
    
    ##CALCULATES THE DIST FROM ELEMENTS WITHIN THE DF TO THE QUERY POINT
    df = df.withColumn('distances', distance_udf(F.col('fv')))
    
    ##SORTS AND RETRIEVES THE TOP-3 RESULTS
    resultSet = df.select("id", "distances").orderBy('distances').limit(k)
    return(resultSet)



##
## TESTING
##


## load the dataset (local)
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("coord_x", FloatType(), True),
    StructField("coord_y", FloatType(), True)])

df = spark.read.csv("/home/weber/UFF-study/coordDF.csv",schema=schema)

##SMALL SANITY CHECK - @PRODUCTION TESTAR LOADING
df = df.na.drop()

## get oq randomly
oq = [np.random.rand(1).round(5)[0],np.random.rand(1).round(5)[0]]

## calling function knn
result = knn(df, oq, 15) # dataframe , query object, k value
result.show()

spark.stop