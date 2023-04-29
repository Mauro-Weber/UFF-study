#!/usr/bin/env python3

from pyspark.sql import SparkSession

from pyspark.sql.functions import greatest
from pyspark.sql.window import Window
from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())

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
    window = Window.partitionBy('spark_partition_id').orderBy("lower_bound")
    df = df.withColumn("next_lb", F.lead("lower_bound").over(window))
    
    return df