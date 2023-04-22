#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:50:42 2023

@author: weber
"""

def laesa_knn(df, oq, k, pivots_list):
    length_df = df.count()
    ## PART_ZERO --> CALCULATE LOWER BOUND 
    lista_columns = []
    
    for i in range(len(pivots_list)):
        df = df.withColumn(f"|d(oq,p{i+1})-d(oi,p{i+1})|", \
            abs (distance.euclidean(oq, pivots_list[i][1]) - \
            F.col(f'distances_oi_pivot{i+1}')))

        lista_columns.append(f"|d(oq,p{i+1})-d(oi,p{i+1})|")

    
    df = df.withColumn("lower_bound", greatest(*[col_name for col_name in lista_columns]))

    ## PART_ONE --> DEFINE FIRST WINDOW 
    df_limit_k = df.limit(k).withColumn("dist_Oq", simpleF(oq)(F.col('fv'))).orderBy("dist_Oq")
    r_laesa = df_limit_k.agg(max('dist_Oq')).collect()[0][0]
    df = df.filter(df["lower_bound"] <= r_laesa)
    
    
    ## PART_TWO --> EXTEND WINDOW AND DROP OVERRUN ROWS
    i=k
    while i < df.count():
        i = min(i+k, df.count())
        # PART_TWO
        
        df_limit_k = df.limit(i).withColumn("dist_Oq", simpleF(oq)(F.col('fv'))).orderBy("dist_Oq").limit(k)
        r_laesa = df_limit_k.agg(max('dist_Oq')).collect()[0][0]
        
        df = df.filter(df["lower_bound"] <= r_laesa)
    
    ## RETURN RESULT SET
    rSet = df_limit_k["id","dist_Oq"]
    #rSet.show()
    drop_rows = length_df - i
    return(rSet, drop_rows)  
