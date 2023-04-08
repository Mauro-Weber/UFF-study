def bridk_simple(df, oqDF, k):
        
    ##CALCULATES THE DIST FROM ELEMENTS WITHIN THE DF TO THE QUERY POINT
    df = df.withColumn('distances', simpleF(oq1)(F.col('fv'))).orderBy('distances')    
    
    # auxiliary variables
    list_result = []
    h = 1
    dropped = 0
    first_run = True
    
    # collect rows from df 
    rows = df.collect()
    
    # for each row in sorted df
    for row in rows:
        
        #get first element 
        if first_run:
            id_influence = row["id"]
            point_influence = row["fv"]
            r_influence = row["distances"]
            list_result.append((row["id"],row["distances"]))
            first_run =  False
            continue
            
        distanceToR = distance.euclidean(point_influence,row["fv"])
    
        if distanceToR > r_influence:

            #GET ELEMENT
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