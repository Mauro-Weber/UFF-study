#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:22:16 2023

@author: weber
"""

# Import SparkSession
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, FloatType, StructType, StructField, IntegerType

## the distance function
from scipy.spatial import distance
## creates the feature vector
from pyspark.ml.feature import VectorAssembler

## start session
spark = SparkSession.builder.appName("SparkKnnJoin").getOrCreate()

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())


## defines query object and 
oq = [(1.0,[0.0,0.0]),(2.0,[1.0,1.0]),(3.0,[0.5,0.5])]
oqColumns = ["idOq","fvOq"]
oqDF = spark.createDataFrame(data=oq, schema = oqColumns)


## load the dataset (local)

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("coord_x", FloatType(), True),
    StructField("coord_y", FloatType(), True)])


df = spark.read.csv("/home/weber/UFF-study/coordDF.csv",schema=schema)

##SMALL SANITY CHECK - @PRODUCTION TESTAR LOADING
df = df.na.drop()

def knnJoin(df, oqDF, k):

    
    ## define the struct for the dimensional feature vector
    cNames = df.columns
    cNames.remove("id")
    assembler = VectorAssembler(
        inputCols=cNames,
        outputCol="fv")
    
    
    ## appends the fv into the dataframe as column
    df = assembler.transform(df)
    
    for i in range(oqDF.count()):
        
        oq = oqDF.filter(oqDF.idOq == i+1).collect()[0][1]
        #COMPUTES THE PAIRWISE DISTANCES
        df_new = df.withColumn(f'distances_oq{i+1}', simpleF(oq)(F.col('fv')))
    
        ##
        ## PYTHON-ish WORKAROUNDS FOR RELATIONAL-LIKE DATAFRAMES
        ## 1- CREATE A TEMPLATE TO STORE THE LIST OF PAIRS (id, distToOq)
        ## 2- STORE THE LIST OF PAIRS INSIDE A COLUMN - CALLED pw_distances
        ##
        templateGroupedList = F.struct(["id", f"distances_oq{i+1}"]).alias("pw_distances")
    
        ## CREATES THE LIST OF PAIRS BY idOq
        df_new = df_new.agg(F.collect_list(templateGroupedList).alias("pw_distances")).withColumn("idOq", F.lit(i+1))
    
        ## SORTS THE LIST OF PAIRS BY DISTANCE
        ## F.expr SQL-LIKE COMMANDS - @TODO CHECK PERFORMANCE - SORTING IS THE DOMINANT COMPLEXITY = ACHO QUE DEVE SER BEM EFICIENTE
        df_new = df_new.withColumn(
          "pw_distances",
          F.expr(
            f"array_sort(pw_distances, (left, right) -> case when left.distances_oq{i+1} < right.distances_oq{i+1} then -1 when left.distances_oq{i+1} > right.distances_oq{i+1} then 1 else 0 end)"
          )
        )
    
        ## FETCHES kNN RESULTS
        ## P.D.: INDEX STARTS WITH 1 FOR SLICING =S
        reSet = df_new.withColumn("kNN", F.slice(F.col("pw_distances"),1,k))
    
       
        
        if i == 0:
            resultSet = reSet
        else:
            
            resultSet = reSet.union(resultSet)
            
    return resultSet


df_final = knnJoin(df, oqDF, 3)

df_final.show()
            
## PRINT kNN RESULT SET (IDS)
df_final.select("idOq", "kNN.id").show()


## PRINT kNN RESULT SET (DISTS)
df_final.select(
    'idOq', *[F.col('kNN')[i].dropFields("id").alias(f'distNN{i+1}') for i in range(3)]
).show()

spark.stop