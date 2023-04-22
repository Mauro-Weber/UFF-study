def bridk_brute(df, oqDF, k):
      
    df = df.withColumn('distances', simpleF(oq1)(F.col('fv'))).orderBy('distances')    
    
    list_result = []
    h = 1
    dropped = 0
    first_run = True
    
    rows = df.collect()
    for row in rows:
         
        if first_run:
            id_influence = row["id"]
            point_influence = row["fv"]
            r_influence = row["distances"]
            list_result.append((row["id"],row["distances"]))
            first_run =  False
            continue
            
        distanceToR = distance.euclidean(point_influence,row["fv"])
    
        if distanceToR > r_influence:

            id_influence = row["id"]
            point_influence = row["fv"]
            r_influence = row["distances"]
            list_result.append((row["id"],row["distances"]))
            h += 1
            
        else:
            dropped += 1
        
        if h == k:
            break
    
    return (list_result, dropped)  
