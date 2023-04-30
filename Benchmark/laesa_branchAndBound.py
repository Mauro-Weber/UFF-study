
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:44:53 2023

@author: weber
"""

from pyspark.sql.types import DoubleType
from scipy.spatial import distance
from reduce_knn import reduce_knn
import pyspark.sql.functions as F
import pandas as pd
import heapq

def distanceF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())


def laesa(df, oq, k):  
    r_laesa = float('inf')
    h = 0
    count = df.count()
    drops = 0
    not_stop_iteration = True
    def process_partition(iterator):
        nonlocal r_laesa, h, drops, not_stop_iteration 
        global_pq = []
        for row in iterator:
            if not_stop_iteration:
                h += 1
                if len(global_pq) < k:
                    heapq.heappush(global_pq, (-(distance.euclidean(oq, row.fv)), row.id))
                else:
                    nextDist = -(distance.euclidean(oq, row.fv))
                    if (nextDist > global_pq[0][0]):
                        heapq.heappush(global_pq, (nextDist, row.id))
                        heapq.heappop(global_pq)   
                        r_laesa = global_pq[0][0]

                    if row.next_lb is None:
                        continue
                    if k < h < count and row.lower_bound >= -(r_laesa):
                        not_stop_iteration = False

        yield global_pq, h
    
    rdd = df.rdd.mapPartitions(process_partition)
    pq = rdd.flatMap(lambda x: x).collect()
    print(pq)
    rslt = reduce_knn(pq, count, k)

    return (rslt[0], rslt[1])