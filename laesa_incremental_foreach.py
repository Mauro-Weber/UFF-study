df = df.orderBy("lower_bound_oq_1")
window = Window.orderBy("lower_bound_oq_1")
df = df.withColumn("next_lb", F.lead("lower_bound_oq_1").over(window))

##
################################################################################
##

def laesa2(df, oq, k): 
    h = 0
    count = df.count()
    drops = 0
    stop = 0
    not_stop_iteration = True
    def process_partition(iterator):
        nonlocal h, drops, not_stop_iteration, stop
        global_pq = []
        new_list = []
        for row in iterator:
            if not_stop_iteration:
                h += 1
                minimo = row.next_lb
                if distance.euclidean(oq, row.fv) < minimo:
                    new_list.append([row.id,distance.euclidean(oq, row.fv)])
                    stop += 1

                else:
                    heapq.heappush(global_pq, (distance.euclidean(oq, row.fv),row.id))
                    
                    if (global_pq[0][0]) < minimo:
                        new_list.append([global_pq[0][1],global_pq[0][0]])
                        heapq.heappop(global_pq) 
                        stop += 1

                if stop > k-1:
                    not_stop_iteration = False
        yield new_list, h
    
    rdd = df.rdd.mapPartitions(process_partition)
    pq = rdd.flatMap(lambda x: x).collect()
    drops = count-pq[1]
    return (pq[0], drops)

##
##############################################################################
##
