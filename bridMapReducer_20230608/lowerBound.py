#!/usr/bin/env python3

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *

from pyspark.sql.functions import spark_partition_id
from pyspark.sql.functions import greatest
from pyspark.sql.window import Window

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())


def lowerBound(df, rowsOq, pivots_list):
    for rowOq in rowsOq:
        print(f"OQ -- {rowOq.id}")
        
        lista_columns = []
            
        for i in range(len(pivots_list)):
            df = df.withColumn(f"|d({rowOq.id},p{i+1})-d(oi,p{i+1})|", \
                F.abs(distance.euclidean(rowOq['fv'], pivots_list[i][1]) - \
                F.col(f'distances_oi_pivot{i+1}')))

            lista_columns.append(f"|d({rowOq.id},p{i+1})-d(oi,p{i+1})|")



        df = df.withColumn(f"lower_bound_{rowOq.id}", greatest(*[col_name for col_name in lista_columns]))

        df = df.withColumn('dummy', F.lit(1))
        window = Window.partitionBy('dummy').orderBy(f"lower_bound_{rowOq.id}")
        df = df.withColumn(f"next_lb_{rowOq.id}", F.lead(f"lower_bound_{rowOq.id}").over(window))
        df = df.drop("dummy")

    return(df)

def repartitionLowerBound(df,oq):
    df_repartitioned = df.repartition(8)
    df_repartitioned = df_repartitioned.withColumn('partition_id', spark_partition_id()).\
                                        sortWithinPartitions(f"lower_bound_{oq[0]}")
    
    df = df_repartitioned

    return(df)