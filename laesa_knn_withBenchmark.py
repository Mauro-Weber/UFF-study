#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 18:55:46 2023

@author: weber
"""

##
## ---------------------------------------------------------------------------
##

# Import SparkSession
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, FloatType, StructType, StructField, IntegerType,  StringType

## the distance function
from scipy.spatial import distance
## creates the feature vector
from pyspark.ml.feature import VectorAssembler
## import numpy
import numpy as np

from pyspark.sql.functions import max, abs

from pyspark.sql.functions import greatest

import time

## start session
spark = SparkSession.builder.appName("SparkLAESAKnn").getOrCreate()




##DEFINES A UDF FROM L2DIST THE DATAFRAME FROM A QUERY SET
distance_udf = F.udf(lambda x,y: float(distance.euclidean(x, y)), DoubleType())

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())



##
## ----------------------------------------------------------------------------
##

def laesa_knn(df, oq, k, pivots_list):
    length_df = df.count()
    ## PART_ZERO --> CALCULATE LOWER BOUND 
    lista_columns = []
    for i in range(len(pivots_list)):
        df = df.withColumn(f"|d(oq,p{i+1})-d(oi,p{i+1})|", \
            abs (distance.euclidean(oq, pivots_list[i][1]) - \
            F.col(f'distances_oi_pivot{i+1}')))

        lista_columns.append(f"|d(oq,p{i+1})-d(oi,p{i+1})|")

    
    
    df = df.withColumn("lower_bound", greatest(*[col_name for col_name in lista_columns]))
    
    
    ## PART_ONE --> DEFINE FIRST WINDOW 
    df_limit_k = df.limit(k).withColumn("dist_Oq", simpleF(oq)(F.col('fv'))).orderBy("dist_Oq")
    r_laesa = df_limit_k.agg(max('dist_Oq')).collect()[0][0]
    df = df.filter(df["lower_bound"] <= r_laesa)
    
    ## PART_TWO --> EXTEND WINDOW AND DROP OVERRUN ROWS
    i=k
    while i < df.count():
        i = min(i+k, df.count())
        # PART_TWO
        
        df_limit_k = df.limit(i).withColumn("dist_Oq", simpleF(oq)(F.col('fv'))).orderBy("dist_Oq").limit(k)
        r_laesa = df_limit_k.agg(max('dist_Oq')).collect()[0][0]
        
        df = df.filter(df["lower_bound"] <= r_laesa)
    
    ## RETURN RESULT SET
    rSet = df_limit_k["id","dist_Oq"]     
    drop_rows = length_df - i
    return(rSet, drop_rows)  

##
##-------------------------------------------------------------------------
##


## 
## SINGLE CALL
##



def singleCall():
    
    ## defines query object and 
    oq = [np.random.rand(1)[0],np.random.rand(1)[0]]
    
    ## defines neighbors 
    k = 5
    
    ## defines dataframe    
    dataframePath = "/home/weber/Documents/coordDF1K.csv"
    
    # define pivots amount
    pivots = 2    
        
    ##
    ## START DATAFRAME CONFIG
    ##
    
    
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("coord_x", FloatType(), True),
        StructField("coord_y", FloatType(), True)])
    
    df = spark.read.csv(dataframePath ,schema=schema)        
    
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
    
            
    # creates random pivots
    pivots_list = [(i+1,[np.random.rand(1)[0],np.random.rand(1)[0]]) for i in range(pivots)]
    
    ## calculates the distance from oi to pivots
    for i in range(len(pivots_list)):
        df = df.withColumn(f'distances_oi_pivot{i+1}', simpleF(pivots_list[i][1])(F.col('fv')))
    
    # measure the execution time of the laesa function
    start_time_laesa = time.time()
    result = laesa_knn(df, oq, k, pivots_list)
    end_time_laesa = time.time()               
    
    result[0].show()
    print(f'number of droped rows: {result[1]}')
    print(f'execution time {end_time_laesa-start_time_laesa}')
    
    return()
            



## 
## BENCHMARK
##

def benchmark():
    
    ## defines query object and 
    oq = [np.random.rand(1)[0],np.random.rand(1)[0]]
    
    ## defines neighbrs amount
    list_k = [5,10]
    
    ## defines size of dataframe    
    list_dfSize = ["/home/weber/Documents/coordDF100.csv"]
                  
    # define pivots amount
    list_pivot = [2]
      
     
    # create a list to save the results
    resultList = []
    # iterative index "id"
    j = 1
    
    
    # for each dataframe
    for df_size in list_dfSize:
        
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("coord_x", FloatType(), True),
            StructField("coord_y", FloatType(), True)])
        
        df = spark.read.csv(df_size ,schema=schema)        
        size = df.count()
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
        
        # for each k on the list of neighbors amount
        for k in list_k:
            
            # for each pivots amount
            for pivots in list_pivot:
                
                # creates random pivots
                pivots_list = [(i+1,[np.random.rand(1)[0],np.random.rand(1)[0]]) for i in range(pivots)]
                
                ## calculates the distance from oi to pivots
                for i in range(len(pivots_list)):
                    df = df.withColumn(f'distances_oi_pivot{i+1}', simpleF(pivots_list[i][1])(F.col('fv')))
                
                # measure the execution time of the laesa function
                start_time_laesa = time.time()
                result = laesa_knn(df, oq, k, pivots_list)
                end_time_laesa = time.time()               
                
                # save results in a tuple
                tuple = ()
                tuple += (j,"laesa_knn", size, k, pivots, end_time_laesa - start_time_laesa, result[1])
            
                # save results tuple in a List            
                resultList.append(tuple)
                j+=1
            
    
    # turn list of results into a dataframe        
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("method", StringType(), True),
        StructField("df_size", IntegerType(), True),
        StructField("k", IntegerType(), True),
        StructField("pivots", IntegerType(), True),
        StructField("time", FloatType(), True),
        StructField("droped_rows", IntegerType(), True)])
    
    df_final = spark.createDataFrame(data = resultList, schema=schema)
    
    df_final.show()
    
    return()
            
# CALLING BENCHMARK
benchmark()

# CALLING SINGLE
singleCall()


spark.stop