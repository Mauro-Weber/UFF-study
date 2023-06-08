#!/usr/bin/env python3

from pyspark.sql.functions import spark_partition_id

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

    df = df.withColumn('dummy', F.lit(1))
    window = Window.partitionBy('dummy').orderBy("lower_bound")
    df = df.withColumn("next_lb", F.lead("lower_bound").over(window))
    df = df.drop("dummy")



    num_partitions = 5 
    df_repartitioned = df.repartition(num_partitions)
    df_repartitioned = df_repartitioned.withColumn('partition_id', spark_partition_id()).\
                                        sortWithinPartitions("lower_bound")
    
    return df_repartitioned