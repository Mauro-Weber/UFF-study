#!/usr/bin/env python3

from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import *

spark = SparkSession.builder.appName("SparkLAESAKnn").getOrCreate()

def treatedDataFrame(df_name):
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("coord_x", FloatType(), True),
        StructField("coord_y", FloatType(), True)])
    
    df = spark.read.csv(df_name ,schema=schema)        
    size = df.count()
    ##SMALL SANITY CHECK - @PRODUCTION TESTAR LOADING
    df = df.na.drop()
    
    ## define the struct for the dimensional feature vector
    cNames = df.columns
    cNames.remove("id")
    assembler = VectorAssembler(
        inputCols=cNames,
        outputCol="fv")
    
    ## appends the fv into the dataframe as column
    df = assembler.transform(df)
    
    return (df,size)