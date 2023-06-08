#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:43:15 2023

@author: weber
"""

from pyspark.sql.types import DoubleType
from scipy.spatial import distance
import pyspark.sql.functions as F

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())

def knn(dataframe, oq, k):       
    
    df = dataframe.withColumn('distances', simpleF(oq)(F.col('fv')))
    resultSet = df.select("id", "distances").orderBy('distances').limit(k)
    
    return(resultSet)

