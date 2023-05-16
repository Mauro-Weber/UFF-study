#!/usr/bin/env python3

from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import *


def treatedDataFrame(df_name, spark):
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("coord_x", FloatType(), True),
        StructField("coord_y", FloatType(), True),
        StructField("coord_z", FloatType(), True),
        StructField("coord_q", FloatType(), True),
        StructField("coord_k", FloatType(), True),
        StructField("coord_l", FloatType(), True),
        StructField("coord_i", FloatType(), True),
        StructField("coord_p", FloatType(), True)])

    
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

    #df.show()
    
    return (df,size)