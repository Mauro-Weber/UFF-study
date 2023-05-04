#!/usr/bin/env python3

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *

distance_udf = F.udf(lambda x,y: float(distance.euclidean(x, y)), DoubleType())

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())

def bridk_simple(df, oq, k):
    df = df.withColumn('distances', simpleF(oq)(F.col('fv'))).orderBy('distances')    
    h = 1
    first_run = True
    influences = []
    rows = df.collect()
    
    for row in rows:
        if first_run:
            influences.append([row["id"], row["fv"], row["distances"]])
            first_run =  False
            continue
        
        teste = True
        
        for val in influences:
            if (val[2] > distance.euclidean(row["fv"], val[1])):
                teste = False

        if teste == False:
            continue
        
        else:
            influences.append([row["id"], row["fv"], row["distances"]])
            h += 1
        
        if h == k:
            return (influences) 
    
    return (influences) 