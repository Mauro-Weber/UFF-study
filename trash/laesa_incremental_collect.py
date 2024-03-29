
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:45:03 2023

@author: weber
"""

from pyspark.sql.types import DoubleType
from scipy.spatial import distance
import pyspark.sql.functions as F
import heapq

def distanceF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())


def laesa(df, oq, k):
    df = df.orderBy("lower_bound_oq_1")
    df_min = df["id","lower_bound_oq_1"]
    
    global_pq = []
    ktop_list = []
    stop = 0

    rows = df.collect()
    for row in rows:
        minimo = df_min.collect()[1][1]
        if distance.euclidean(oq, row.fv) < minimo:
            ktop_list.append([row.id,distance.euclidean(oq, row.fv)])
            stop += 1

        else:
            heapq.heappush(global_pq, (distance.euclidean(oq, row.fv),row.id))

            if (global_pq[0][0]) < minimo:
                ktop_list.append([global_pq[0][1],global_pq[0][0]])
                heapq.heappop(global_pq) 
                stop += 1

        if stop > k-1:
            break

        df_min = df_min.filter(F.col("id") != row.id)

    return(ktop_list)
