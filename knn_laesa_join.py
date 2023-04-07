def laesa_join3(df, oqDF, k):  
    r_laesa = float('inf')
    h = 0
    count = df.count()
    stop_iteration = True
    dict_results = {}
    for oq in rows_oqDF: 
        
        df_oq = df.orderBy(df[f"lower_bound_oq_{int(oq.idOq)}"])
        def process_partition(iterator):
            nonlocal r_laesa, h, stop_iteration 
            global_pq = []
            for row in iterator:
                if stop_iteration:
                    h += 1
                    if len(global_pq) < k:
                        heapq.heappush(global_pq, (-(distance.euclidean(oq.fvOq, row.fv)), row.id))
                    else:
                        nextDist = -(distance.euclidean(oq.fvOq, row.fv))
                        if (nextDist > global_pq[0][0]): 
                            heapq.heappush(global_pq, (nextDist, row.id))
                            heapq.heappop(global_pq) 
                            r_laesa = global_pq[0][0]
                    if k < h < count and row[f"lower_bound_oq_{int(oq.idOq)}"] >= -(r_laesa):
                        stop_iteration = False
            yield global_pq, count-h
        rdd = df_oq.rdd.mapPartitions(process_partition)
        pq = rdd.flatMap(lambda x: x).collect()
    
        dict_results[oq.idOq] = pq

    return dict_results
