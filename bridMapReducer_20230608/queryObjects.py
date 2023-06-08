#!/usr/bin/env python3

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType()) 


def queryObjectList(df): 
    objectsList = []
    def getList(iterator):
        nonlocal objectsList
        for row in iterator:
            objectTuple = tuple([row.id,row.fv])
            objectsList.append(objectTuple)         

        yield objectsList

    objectListResult = df.rdd.mapPartitions(getList)

    return(objectListResult)