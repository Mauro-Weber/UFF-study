#!/usr/bin/env python3

from scipy.spatial import distance
from bridk_reduce import bridk_reduce

import heapq


import time
                 

def bridk_incremental(df, oq, k): 
    h = 0
    count = df.count()
    drops = 0
    add_neighbor = 0
    not_stop_iteration = True
    queue_pq = [(float('inf'), 0, [0.0, 0.0])]
    influence_list = []
    start_iterator = False

    def process_partition(iterator):
        nonlocal h, k, start_iterator, not_stop_iteration, add_neighbor, queue_pq, influence_list

        for row in iterator:            
            start_iterator = True
            first_teste_infl = True
            second_teste_infl = True

            if not_stop_iteration == True:

                if add_neighbor >= k:
                    not_stop_iteration = False
                    start_iterator == False
                    continue
                
                h += 1
                for val in influence_list:
                    if (val[1] > distance.euclidean(val[2], row.fv)):
                        first_teste_infl = False  
                
                if first_teste_infl == False:
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
                                second_teste_infl = False  

                        if second_teste_infl == False:
                            heapq.heappop(queue_pq)
                            continue
                        
                        influence_list.append([queue_pq[0][1],\
                                               distance.euclidean(oq, queue_pq[0][2]),\
                                               queue_pq[0][2]])
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
                                           queue_pq[0][2]])
                    heapq.heappop(queue_pq) 
                    add_neighbor += 1    

        yield influence_list, h

    
    
    # start_time = time.time()

    rdd = df.rdd.mapPartitions(process_partition)
    pq = rdd.flatMap(lambda x: x).collect()

    # end_time = time.time()

    # print(f'TEMPO BRIDK INCR {end_time - start_time}')





    # start_time = time.time()

    result = bridk_reduce(pq,k)
    # end_time = time.time()

    # print(f'TEMPO k={k} REDUCE BRIDK INCR {end_time - start_time}')

    
    drops = count-result[1]
    
    return (result[0], drops)
