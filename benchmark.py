#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:36:27 2023

@author: weber
"""

##
## IMPORTING SESSION
##
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

from pyspark.sql.functions import col

import time

import pyspark.sql.functions as F

from statistics import mode

import matplotlib.pyplot as plt

from functools import reduce

from pyspark.ml.linalg import Vectors

import numpy as np

import heapq


## start session
spark = SparkSession.builder.appName("SparkLAESAKnn").getOrCreate()




##DEFINES A UDF FROM L2DIST THE DATAFRAME FROM A QUERY SET
distance_udf = F.udf(lambda x,y: float(distance.euclidean(x, y)), DoubleType())

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())

##
###############################################################################
##


##
## FUNCTIONS SESSION
##


## KNN FUNCTION

def knn(dataframe, oq, k):       
    
    ##CALCULATES THE DIST FROM ELEMENTS WITHIN THE DF TO THE QUERY POINT
    df = dataframe.withColumn('distances', simpleF(oq)(F.col('fv')))
    
    ##SORTS AND RETRIEVES THE TOP-K RESULTS
    resultSet = df.select("id", "distances").orderBy('distances').limit(k)
    return(resultSet)


## LAESA KNN FUNCTION

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
    #rSet.show()
    drop_rows = length_df - i
    return(rSet, drop_rows)  

##
##-------------------------------------------------------------------------
##

##
## LAESA KNN WITH COLLECT() AND QUEUE
##

def laesa_queue(df, oq, k): 
    pq = []
    r_laesa = float('inf')
    rows = df.collect()
    h=0

    for row in rows: 
        h += 1
        if (len(pq) < k):
            heapq.heappush(pq, (-(distance.euclidean(oq, row.fv)),row.id)) 
        else:
            nextDist = -(distance.euclidean(oq, row.fv))
            if (nextDist > pq[0][0]): 
                heapq.heappush(pq, (nextDist, row.id))
                heapq.heappop(pq) 
                r_laesa = pq[0][0]
        if (k < h < df.count() and row.lower_bound >= -(r_laesa)):
            break
    drops = df.count()-h
    return(pq, drops)


##
##############################################################################
##

##
## LAESA KNN WITH toLocalIterator() AND QUEUE
##

def laesa_queue_1(df, oq, k): 
    pq = []
    r_laesa = float('inf')
    rows = df.rdd.toLocalIterator()
    h=0
    for row in rows: 
        h += 1
        if (len(pq) < k):
            heapq.heappush(pq, (-(distance.euclidean(oq, row.fv)),row.id)) 
        else:
            nextDist = -(distance.euclidean(oq, row.fv))
            if (nextDist > pq[0][0]): 
                heapq.heappush(pq, (nextDist, row.id))
                heapq.heappop(pq) 
                r_laesa = pq[0][0]
        if (k < h < df.count() and row.lower_bound >= -(r_laesa)):
            break
    drops = df.count()-h
    return(pq, drops)


##
##############################################################################
##


##
## DATAFRAME CONFIG
##


## 
## BENCHMARK
##

def benchmark():
    
    ## defines query object and 
    oq = [np.random.rand(1)[0],np.random.rand(1)[0]]
    
    ## defines neighbrs amount
    list_k = [10]
    
    ## defines size of dataframe    
    list_dfSize = ["/home/weber/Documents/coordDF100.csv"]#,"/home/weber/Documents/coordDF1K.csv","/home/weber/Documents/coordDF10K.csv","/home/weber/Documents/coordDF100K.csv"]
                  
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
        lista = []
        length_df = df.count()
        
        # for each k on the list of neighbors amount
        for k in list_k:
            
            # for each pivots amount
            for pivots in list_pivot:
                
                lista_columns = []
                
                # creates random pivots
                pivots_list = [(i+1,[np.random.rand(1)[0],np.random.rand(1)[0]]) for i in range(pivots)]
                
                ## calculates the distance from oi to pivots
                for i in range(len(pivots_list)):
                    df = df.withColumn(f'distances_oi_pivot{i+1}', simpleF(pivots_list[i][1])(F.col('fv')))
                    
                    
                for i in range(len(pivots_list)):
                    df = df.withColumn(f"|d(oq,p{i+1})-d(oi,p{i+1})|", \
                        abs (distance.euclidean(oq, pivots_list[i][1]) - \
                        F.col(f'distances_oi_pivot{i+1}')))

                    lista_columns.append(f"|d(oq,p{i+1})-d(oi,p{i+1})|")



                df = df.withColumn("lower_bound", greatest(*[col_name for col_name in lista_columns]))
                df = df.orderBy(df["lower_bound"])
                
                
                # measure the execution time of the laesa function
                start_time_knn = time.time()
                result3 = knn(df, oq, k)
                end_time_knn = time.time()  

                # save results in a tuple
                tuple = ()
                tuple += (j,"knn", size, k, pivots, end_time_knn - start_time_knn, 0)
            
                # save results tuple in a List            
                resultList.append(tuple)
                j+=1 
                
                
                # measure the execution time of the laesa function
                start_time_laesa_knn = time.time()
                result1 = laesa_knn(df, oq, k, pivots_list)
                end_time_laesa_knn = time.time()  

                # save results in a tuple
                tuple = ()
                tuple += (j,"laesa_knn_withColumn", size, k, pivots, end_time_laesa_knn - start_time_laesa_knn, result1[1])
            
                # save results tuple in a List            
                resultList.append(tuple)
                j+=1 
                
                
                # measure the execution time of the laesa function
                start_time_queue = time.time()
                result4 = laesa_queue(df, oq, k)
                end_time_queue = time.time()  

                # save results in a tuple
                tuple = ()
                tuple += (j,"laesa_queue_collect", size, k, pivots, end_time_queue - start_time_queue, result4[1])
            
                # save results tuple in a List            
                resultList.append(tuple)
                j+=1 
                
                # measure the execution time of the laesa function
                start_time_queue = time.time()
                result5 = laesa_queue_1(df, oq, k)
                end_time_queue = time.time()  

                # save results in a tuple
                tuple = ()
                tuple += (j,"laesa_queue_toLocalIterator", size, k, pivots, end_time_queue - start_time_queue, result5[1])
            
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
    
    return(df_final, lista)


##
## CALLING 
##
            
# CALLING BENCHMARK
lista = benchmark()







##
###############################################################################
##


##
## CREATES BENCHMARK
##


## df          --> dataframe
## algorithm   --> query algorithms
## method      --> pivots selection method 
## n_pivots    --> pivots amount
## n_neighbors --> neighbors amount

def benchmark(df, algorithm, method, n_pivots, n_neighbors):
    
    return()
    
