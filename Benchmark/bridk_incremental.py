#!/usr/bin/env python3

from scipy.spatial import distance
from functools import reduce
import heapq
                 

def bridk_incremental(df, oq, k): 
    h = 0
    count = df.count()
    drops = 0
    add_neighbor = 0
    not_stop_iteration = True
    queue_pq = [(float('inf'), 0, [0.0, 0.0])]
    influence_list = []
    def process_partition(iterator):
        nonlocal h, k, count, not_stop_iteration, add_neighbor, queue_pq, influence_list
        for row in iterator:
            
            teste = True

            if not_stop_iteration == True:

                if add_neighbor >= k:
                    count = count - h
                    not_stop_iteration = False
                    continue
                
                h += 1
                for val in influence_list:
                    if (val[1] > distance.euclidean(val[2], row.fv)):
                        teste = False  
                
                if teste == False:
                    continue
                
                min_lower_bound = row.next_lb
                
                if min_lower_bound is None:
                    continue
                
                if (distance.euclidean(oq, row.fv) < min_lower_bound) and (distance.euclidean(oq, row.fv) < queue_pq[0][0]):
                    influence_list.append([row.id,distance.euclidean(oq, row.fv),row.fv])
                    add_neighbor += 1

                else:
                    heapq.heappush(queue_pq, (distance.euclidean(oq, row.fv),row.id,row.fv))
                    if (queue_pq[0][0]) < min_lower_bound:
                        for val in influence_list:
                            if (val[1] > distance.euclidean(val[2], queue_pq[0][2])):
                                heapq.heappop(queue_pq) 
                                teste = False  

                        if teste == False:
                            continue
                        
                        influence_list.append([queue_pq[0][1],\
                                               distance.euclidean(oq, queue_pq[0][2]),\
                                               queue_pq[0][2]])
                        heapq.heappop(queue_pq) 
                        add_neighbor += 1
        yield influence_list, h

    rdd = df.rdd.mapPartitions(process_partition)
    pq = rdd.flatMap(lambda x: x).collect()
    drops = count-pq[1]
    return (pq[0], drops)
