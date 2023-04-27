#!/usr/bin/env python3

from pyspark.sql import SparkSession

from bridk_incremental import bridk_incremental
from fechoConvexo import FechoConv
from bridk_simple import bridk_simple
from MaxVariance import MaxVariance


from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import greatest
from pyspark.sql.window import Window
from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np
import time


spark = SparkSession.builder.appName("SparkLAESAKnn").getOrCreate()

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())


def treatDataFrame(df_name):
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("coord_x", FloatType(), True),
        StructField("coord_y", FloatType(), True)])
    
    df = spark.read.csv(df_name ,schema=schema)        
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
    
    return (df,size)


def lowerBound_dataFrame(df, oq, pivots_list):
    
    lista_columns = []
    
    ## calculates the distance from oi to pivots
    for i in range(len(pivots_list)):
        df = df.withColumn(f'distances_oi_pivot{i+1}', simpleF(pivots_list[i][1])(F.col('fv')))
        
        
    for i in range(len(pivots_list)):
        df = df.withColumn(f"|d(oq,p{i+1})-d(oi,p{i+1})|", \
            F.abs(distance.euclidean(oq, pivots_list[i][1]) - \
            F.col(f'distances_oi_pivot{i+1}')))

        lista_columns.append(f"|d(oq,p{i+1})-d(oi,p{i+1})|")



    df = df.withColumn("lower_bound", greatest(*[col_name for col_name in lista_columns]))
    df = df.orderBy(df["lower_bound"])
    
    df = df.orderBy("lower_bound")
    window = Window.orderBy("lower_bound")
    df = df.withColumn("next_lb", F.lead("lower_bound").over(window))
    
    return df

def main():
    
    
    oq = [0.241,0.845]
    
    oq2 = [(1.0,[0.241,0.845])]
    oqColumns = ["idOq","fvOq"]
    oqDF = spark.createDataFrame(data=oq2, schema = oqColumns)
    oqDF.show()
    
    
    list_k = [10]
    
    list_dfNames = ["/home/weber/Documents/coordDF10K.csv"]#,\
                   #"/home/weber/Documents/coordDF1K.csv",\
                   #"/home/weber/Documents/coordDF10K.csv",\
                   #"/home/weber/Documents/coordDF100K.csv"]
                  
    list_pivot = [3,5]
      
     
    # create a list to save the results
    resultList = []
    # iterative index "id"
    j = 1
    
    
    # for each dataframe
    for df_name in list_dfNames:
        
        
        treatR = treatDataFrame(df_name)
        df = treatR[0]
        size = treatR[1]
        
        
            
        # for each pivots amount
        for pivots in list_pivot:
            
            pivots_method = ["random","maxVariance","fechoConvexo"]
            
            for method in pivots_method:
                
                if method == "random":
                    #pivots_list = [(i+1,[np.random.rand(1)[0],np.random.rand(1)[0]]) for i in range(pivots)]
                    random_pivots_list = [(i+1,[np.random.rand(1)[0],np.random.rand(1)[0]]) for i in range(pivots)]
                    pivots_list = random_pivots_list
                
                elif method == "maxVariance":
                    max_lst = MaxVariance(df, pivots)
                    maxVar_pivots_list = max_lst.find_max_variance()
                    pivots_list = maxVar_pivots_list
                 
                
                elif method == "fechoConvexo":
                    fech_conv = FechoConv(df, pivots)
                    fech_pivots_lst = fech_conv.return_pivots()
                    pivots_list = fech_pivots_lst
                    
                   
                # for each k on the list of neighbors amount
                for k in list_k:
                
                    df = lowerBound_dataFrame(df, oq, pivots_list)
                                    
                    start_time_bridk_simple = time.time()
                    result3 = bridk_simple(df, oq, k)
                    end_time_bridk_simple = time.time()  
                    
                    tuple = ()
                    tuple += (j,"bridk_simple", size, k, str(oq), pivots, method, end_time_bridk_simple - start_time_bridk_simple, 0)
                    resultList.append(tuple)
                    j+=1 
                    
                    start_time_bridk_incremental = time.time()
                    result4 = bridk_incremental(df, oq, k)
                    end_time_bridk_incremental = time.time()  

                    tuple = ()
                    tuple += (j,"bridk_incremental", size, k, str(oq), pivots, method, end_time_bridk_incremental - start_time_bridk_incremental, result4[1])
                    resultList.append(tuple)
                    j+=1 
                    
    # turn list of results into a dataframe        
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("method", StringType(), True),
        StructField("df_size", IntegerType(), True),
        StructField("k", IntegerType(), True),
        StructField("oq", StringType(), True),
        StructField("qnt_pivots", IntegerType(), True),
        StructField("select_pivots_method", StringType(), True),
        StructField("time", FloatType(), True),
        StructField("droped_rows", IntegerType(), True)])
    
    df_final = spark.createDataFrame(data = resultList, schema=schema)
    
    df_final.show()
    
    return(df_final)
            

if __name__ == '__main__':
    main()