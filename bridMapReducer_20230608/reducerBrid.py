#!/usr/bin/env python3

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *
import heapq
import time

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType()) 


def reducerBrid(df, oq, k): 
    
    h_reduce = 0  
    start_iterator_reduce = True
    not_stop_iteration_reduce = True
    add_neighbor_reduce = 0 
    queue_pq_reduce = [(float('inf'), 0, [0.0, 0.0])]
    influence_list_reduce = []

    def reducer(iterator):
        nonlocal h_reduce, k, start_iterator_reduce, not_stop_iteration_reduce, add_neighbor_reduce, queue_pq_reduce, influence_list_reduce

        h = h_reduce 
        start_iterator = start_iterator_reduce  
        not_stop_iteration = not_stop_iteration_reduce
        add_neighbor = add_neighbor_reduce
        queue_pq = queue_pq_reduce
        influence_list = influence_list_reduce

        for row in iterator:        
            start_iterator = True
            first_teste_infl = True
            second_teste_infl = True
            if not_stop_iteration == True:

                if add_neighbor >= k:
                    not_stop_iteration = False
                    start_iterator == False
                    
                    yield influence_list
                    return
                
                h += 1

                for val in influence_list:
                    if (val[1] > distance.euclidean(val[2], row[2])):
                        first_teste_infl = False  

                if first_teste_infl == False:
                    continue
                
                min_lower_bound = row[3]
                
                if min_lower_bound is None:
                    continue
                
                if (distance.euclidean(oq, row[2]) < min_lower_bound) and (distance.euclidean(oq, row[2]) < queue_pq[0][0]):
                    influence_list.append([row[0],distance.euclidean(oq, row[2]),row[2],row[3]])
                    add_neighbor += 1

                else:
                    heapq.heappush(queue_pq, (distance.euclidean(oq, row[2]),row[0],row[2],row[3]))
                    if (queue_pq[0][0]) < min_lower_bound:
                        for val in influence_list:
                            if (val[1] > distance.euclidean(val[2], queue_pq[0][2])):
                                second_teste_infl = False  

                        if second_teste_infl == False:
                            heapq.heappop(queue_pq)
                            continue
                        
                        influence_list.append([queue_pq[0][1],\
                                               distance.euclidean(oq, queue_pq[0][2]),\
                                                queue_pq[0][2],\
                                                    queue_pq[0][3]])
                        heapq.heappop(queue_pq) 
                        add_neighbor += 1
        
        if start_iterator:

            while (add_neighbor < k) and (len(queue_pq) > 1):
                infl_test = True
                for val in influence_list:
                    if (val[1] > distance.euclidean(val[2], queue_pq[0][2])):
                        infl_test = False 

                if infl_test == False:
                    heapq.heappop(queue_pq) 

                else:
                    influence_list.append([queue_pq[0][1],\
                                           distance.euclidean(oq, queue_pq[0][2]),\
                                           queue_pq[0][2],\
                                            queue_pq[0][3]])
                    heapq.heappop(queue_pq) 
                    add_neighbor += 1    

        yield influence_list

    dfReducer = df.flatMap(lambda x: [x[0]]).flatMap(lambda x: x).coalesce(1).map(lambda x: tuple(x))
    dfReducerSort = dfReducer.sortBy(lambda x: x[3])
    reducerResult = dfReducerSort.mapPartitions(reducer)

    return(reducerResult)