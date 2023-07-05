#!/usr/bin/env python3

from pyspark.sql import SparkSession

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np
import time

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())

if __name__ == '__main__':

    spark = SparkSession.builder \
        .getOrCreate()


    df = spark.read.parquet("./oqdataframe_1M_.parquet") 
    df.show()
    
    
