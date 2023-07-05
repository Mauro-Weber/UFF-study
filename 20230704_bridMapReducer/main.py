#!/usr/bin/env python3

from pyspark.sql import SparkSession

from pyspark.sql.window import Window
from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np
import time

from reducerBrid import reducerBrid
from mapBrid import mapBrid

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("SparkTest") \
        .getOrCreate()

    #input parameters
    list_k = list(range(5, 101, 5))  # Number of neighbors   
    list_pivot = [2] # Number of pivots
    
    # create indices for resultDataset
    j = 0
    resultList = []

    # load dataframes 
    df = spark.read.parquet("Documents/mapReducer2/20230704/dataframes/dataframe_1M_.parquet")
    oqdf = spark.read.parquet("Documents/mapReducer2/20230704/dataframes/oqdataframe_1M_.parquet")

    # cartesian product of oqDF and DF
    df = oqdf.crossJoin(df)

    # Obter os valores das colunas "id_oq" e "fv_oq" como uma lista de tuplas
    list_oq = oqdf.select("id_oq", "fv_oq").rdd.map(tuple).collect()
    
    i_values = [1, 2]
    for i in i_values:
        dist_diff_col_name = f"lower_bound_{i}"
        df = df.withColumn(dist_diff_col_name, F.abs(F.col(f"dist_oq_pivot{i}") - F.col(f"dist_pivot{i}")))

    df = df.withColumn("lower_bound", F.greatest(*[F.col(f"lower_bound_{i}") for i in i_values]))

    for oq in list_oq:

        new_df = df.filter(F.col("id_oq") == oq[0])
        new_df = new_df.sort("lower_bound")
        new_df  = new_df.withColumn("next_lower_bound", F.lead("lower_bound").over(Window.orderBy("lower_bound")))
        new_df = new_df.repartition(11)

        t_start = time.time()

        for k in list_k:

            rst_map = mapBrid(new_df, k)
            t_stop = time.time()
            mapTime = t_stop - t_start
            r_start = time.time()
            rst_reducer = reducerBrid(rst_map, oq[1], k)
            r_stop = time.time()
            reducerTime = r_stop - r_start

            full = mapTime + reducerTime
            tuple_s = ()
            tuple_s += (j, k, str(oq[0]), full, mapTime, reducerTime)
            resultList.append(tuple_s)
            j+=1


    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("k", IntegerType(), True),
        StructField("oq", StringType(), True),
        StructField("full_function_time", FloatType(), True),
        StructField("map_time", FloatType(), True),
        StructField("reducer_time", FloatType(), True)
    ])

    df_final = spark.createDataFrame(data = resultList, schema=schema)

    p = 0
    avgList = []

    for k in range(5, 101, 5):
        media_tempos = df_final.filter(df_final.k == k).select(F.avg("full_function_time")).first()[0]

        tuple_avg = ()
        tuple_avg += (p, k, media_tempos)
        avgList.append(tuple_avg)
        p+=1

    schema_avg = StructType([
        StructField("id", IntegerType(), True),
        StructField("k", IntegerType(), True),
        StructField("execution_time", StringType(), True),
    ])

    df_avg = spark.createDataFrame(data = avgList, schema=schema_avg)
    df_avg.show()

    spark.stop()

