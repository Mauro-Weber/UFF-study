##
## LAESA KNN WITH foreach() AND A QUEUE
##

def laesa_queue5(df, oq, k):  
    r_laesa = float('inf')
    h = 0
    count = df.count()
    drops = 0
    stop_iteration = True
    def process_partition(iterator):
        nonlocal r_laesa, h, drops, stop_iteration 
        global_pq = []
        for row in iterator:
            if stop_iteration:
                h += 1
                if len(global_pq) < k:
                    heapq.heappush(global_pq, (-(distance.euclidean(oq, row.fv)), row.id))
                else:
                    nextDist = -(distance.euclidean(oq, row.fv))
                    if (nextDist > global_pq[0][0]): 
                        heapq.heappush(global_pq, (nextDist, row.id))
                        heapq.heappop(global_pq) 
                        r_laesa = global_pq[0][0]
                if k < h < count and row.lower_bound >= -(r_laesa):
                    stop_iteration = False
        yield global_pq, h
    rdd = df.rdd.mapPartitions(process_partition)
    pq = rdd.flatMap(lambda x: x).collect()
    drops = count-pq[1]
    return (pq, drops)

##
##############################################################################
##

