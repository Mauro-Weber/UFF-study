#!/usr/bin/env python3

from pyspark.sql import SparkSession
import pyspark

from lowerBound_dataframe import lowerBound_dataFrame
from laesa_branchAndBound import laesa
from bridk_incremental import bridk_incremental
from bridk_simple_mapPart import bridk_simple as bridk_simple_2
from treatedDataframe import treatedDataFrame
from bridk_simple import bridk_simple
from knn_brute import knn

from MaxVariance import MaxVariance
from fechoConvexo import FechoConv

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np
import time


spark = SparkSession.builder \
    .appName("SparkTest") \
    .config("spark.executor.cores", "8") \
    .config("spark.task.cpus", "8") \
    .getOrCreate()

print("Número de cores configurados:", spark.conf.get("spark.executor.cores"))
print("Número de CPUs por tarefa configurados:", spark.conf.get("spark.task.cpus"))



#print(spark.sparkContext.getConf().get('spark.master'))

# if 'yarn' in spark.sparkContext.getConf().get('spark.master'):
#     print("O código está sendo executado no ambiente YARN.")
# else:
#     print("O código está sendo executado localmente.")



def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())


def main():
    
    
    #oq = [0.241,0.845]
    
    oq2 = [(1.0,[0.241,0.845])]
    oqColumns = ["idOq","fvOq"]
    oqDF = spark.createDataFrame(data=oq2, schema = oqColumns)
    
    
    list_k = list(range(5, 26, 5))
    
    #list_dfNames = ["/home/weber/Documents/coordDF10K.csv"]#,\
                   #"/home/weber/Documents/coordDF1K.csv",\
                   #"/home/weber/Documents/coordDF10K.csv",\
                   #"/home/weber/Documents/coordDF100K.csv"]
                  
    list_pivot = [2]
      
    resultList = []
    j = 1
    
    treatR = treatedDataFrame("/home/weber/Documents/coordDF100K_8D.csv", spark)
    df = treatR[0]
    oqDF = df.sample(False, 0.000024, seed=12)
    df = df.subtract(oqDF) 
    size = treatR[1]

    oqDF.show()

    rowsOq = oqDF.collect()
   


    
    for pivots in list_pivot:
        pivots_method = ["random"]#,"maxVariance","fechoConvexo"]
        
        start_time_pivot_select = time.time()

        for method in pivots_method:
            if method == "random":
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
            
            end_time_pivot_select = time.time()

            for rowOq in rowsOq:
                print(f"OQ -- {rowOq.id}")
                for k in list_k:
                
                    df = lowerBound_dataFrame(df, rowOq.fv, pivots_list)
                                    
                    start_time_bridk_simple = time.time()
                    result3 = bridk_simple_2(df, rowOq.fv, k)
                    end_time_bridk_simple = time.time()  

                    tuple = ()
                    tuple += (j,"bridk_simple", size, k, str(rowOq.id), pivots, method, end_time_pivot_select -start_time_pivot_select, end_time_bridk_simple - start_time_bridk_simple, 0)
                    resultList.append(tuple)
                    j+=1 
                    
                    start_time_bridk_incremental = time.time()
                    result4 = bridk_incremental(df, rowOq.fv, k)
                    end_time_bridk_incremental = time.time()  

                    tuple = ()
                    tuple += (j,"bridk_incremental", size, k, str(rowOq.id), pivots, method, end_time_pivot_select -start_time_pivot_select, end_time_bridk_incremental - start_time_bridk_incremental, result4[1])
                    resultList.append(tuple)
                    j+=1 


                    # start_time_laesa = time.time()
                    # result5 = laesa(df, rowOq.fv, k)
                    # end_time_laesa = time.time()  

                    # tuple = ()
                    # tuple += (j,"laesa_knn", size, k, str(rowOq.id), pivots, method, end_time_pivot_select -start_time_pivot_select, end_time_laesa - start_time_laesa, result5[1])
                    # resultList.append(tuple)
                    # j+=1 


                    start_time_knn = time.time()
                    result6 = knn(df, rowOq.fv, k)
                    end_time_knn = time.time()  

                    tuple = ()
                    tuple += (j,"brute_knn", size, k, str(rowOq.id), pivots, method, end_time_pivot_select -start_time_pivot_select, end_time_knn - start_time_knn, 0)
                    resultList.append(tuple)
                    j+=1 
                         
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("method", StringType(), True),
        StructField("df_size", IntegerType(), True),
        StructField("k", IntegerType(), True),
        StructField("oq", StringType(), True),
        StructField("qnt_pivots", IntegerType(), True),
        StructField("select_pivots_method", StringType(), True),
        StructField("time_select_pivots_method", FloatType(), True),
        StructField("execution_time", FloatType(), True),
        StructField("droped_rows", IntegerType(), True)])
    
    df_final = spark.createDataFrame(data = resultList, schema=schema)
    
    df_final.show()

    df_final.coalesce(1).write.csv("/home/weber/Documents/df100K_core24_cpu8.csv",header=True, mode="overwrite")

    spark.stop()

    return(df_final)
            

if __name__ == '__main__':
    main()