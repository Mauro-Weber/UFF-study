#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:49:10 2023

@author: weber
"""

def laesa_queue_toLocalIterator(df, oq, k): 
    pq = []
    r_laesa = float('inf')
    rows = df.rdd.toLocalIterator()
    h=0
    for row in rows: 
        h += 1
        if (len(pq) < k):
            heapq.heappush(pq, (-(distance.euclidean(oq, row.fv)),row.id)) 
        else:
            nextDist = -(distance.euclidean(oq, row.fv))
            if (nextDist > pq[0][0]): 
                heapq.heappush(pq, (nextDist, row.id))
                heapq.heappop(pq) 
                r_laesa = pq[0][0]
        if (k < h < df.count() and row.lower_bound >= -(r_laesa)):
            break
    drops = df.count()-h
    return(pq, drops)