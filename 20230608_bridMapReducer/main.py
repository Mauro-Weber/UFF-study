#!/usr/bin/env python3

from pyspark.sql import SparkSession

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np

from lowerBound import repartitionLowerBound
from vectorCreation import vectorCreation
from queryObjects import queryObjectList
from reducerBrid import reducerBrid
from lowerBound import lowerBound
from mapBrid import mapBrid

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("SparkTest") \
        .getOrCreate()

    #input parameters
    list_k = list(range(10, 11, 5))  # Number of neighbors   
    list_pivot = [2] # Number of pivots


    # read dataframe 
    schema = StructType([
        StructField("id", IntegerType(), True),
        *[
            StructField(f"coord_{i+1}", FloatType(), True) for i in range(2)
        ]
    ])

    df = spark.read.csv("/home/weber/Documents/coordDF10K.csv" ,schema=schema) 


    # create a column with a vector containing all coordinates
    df = vectorCreation(df)


    # sample of object query
    oqDF = df.sample(False, 0.0001, seed=32)
    df = df.subtract(oqDF) 

    print(oqDF.count())

    # listing all query objects 
    testeQBL = queryObjectList(oqDF)
    rowsOq = oqDF.collect()
    list_oq = [(row.id,row.fv) for row in rowsOq]


    # create random pivots
    random_pivots_list = [(i+1, [np.random.rand(1)[0] for _ in range(2)]) for i in range(2)]
    pivots_list = random_pivots_list


    # calculates the distance from oi to pivots
    for i in range(len(pivots_list)):
        df = df.withColumn(f'distances_oi_pivot{i+1}', simpleF(pivots_list[i][1])(F.col('fv')))

    df = lowerBound(df, rowsOq, pivots_list)

    for oq in list_oq:

        df = repartitionLowerBound(df,oq)

        for k in list_k:
            mapResult = mapBrid(df, oq[1], k, oq[0])
            reducerResult = reducerBrid(mapResult, oq[1], k)
            print(reducerResult.flatMap(lambda x: x).collect())