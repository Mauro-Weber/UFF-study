#!/usr/bin/env python3

from pyspark.ml.feature import VectorAssembler


def vectorCreation(df):
    ## define the struct for the dimensional feature vector
    cNames = df.columns
    cNames.remove("id")
    assembler = VectorAssembler(
        inputCols=cNames,
        outputCol="fv")

    ## appends the fv into the dataframe as column
    df = assembler.transform(df)

    return(df)