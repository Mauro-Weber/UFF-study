#!/usr/bin/env python3

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *
import heapq
import time


def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())


def mapBrid(df, k):

    h = 0    # Quantity of rows read   
    add_neighbor = 0   # Number of chosen Neighbors
    not_stop_iteration = True   # Condition to not stop iteration
    queue_pq = [(float('inf'), 0, [0.0, 0.0])]  # queue of objects
    influence_list = []  # influence list of the chosen neighbors 
    start_iterator = False  # condition to start the second execution when the the k neighbors weren't found

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

                    yield influence_list, h
                    return

                h += 1

                for val in influence_list:
                    if (val[1] > distance.euclidean(val[2], row.fv)):
                        first_teste_infl = False

                if first_teste_infl == False:
                    continue

                min_lower_bound = row.next_lower_bound

                if min_lower_bound is None:
                    continue

                if (distance.euclidean(row.fv_oq, row.fv) < min_lower_bound) and (distance.euclidean(row.fv_oq, row.fv) < queue_pq[0][0]):
                    influence_list.append([row.id,distance.euclidean(row.fv_oq, row.fv),row.fv,row.next_lower_bound])
                    add_neighbor += 1

                else:
                    heapq.heappush(queue_pq, (distance.euclidean(row.fv_oq, row.fv),row.id,row.fv,row.next_lower_bound))
                    if (queue_pq[0][0]) < min_lower_bound:
                        for val in influence_list:
                            if (val[1] > distance.euclidean(val[2], queue_pq[0][2])):
                                second_teste_infl = False

                        if second_teste_infl == False:
                            heapq.heappop(queue_pq)
                            continue

                        influence_list.append([queue_pq[0][1],\
                                               distance.euclidean(row.fv_oq, queue_pq[0][2]),\
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
                                           distance.euclidean(row.fv_oq, queue_pq[0][2]),\
                                           queue_pq[0][2],\
                                            queue_pq[0][3]])
                    heapq.heappop(queue_pq)
                    add_neighbor += 1

        yield influence_list, h

    df_map = df.rdd.mapPartitions(process_partition)
    return (df_map)
