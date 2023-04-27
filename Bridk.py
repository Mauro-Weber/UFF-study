def bridk(df, oqDF, k):
    df = df.withColumn("index_row", monotonically_increasing_id())
    
    # auxiliaries
    
    coordenadas = []
    influences = []
    aux = []
    influence_dictio = {}
    x=0
   
    # end-aux
    
    rows_oqDF = oqDF.collect()
    for oq in rows_oqDF:
        
        df = df.withColumn("distance_oq", when((df.index_row < k),simpleF(oq.fvOq)(F.col('fv'))).otherwise(None))

        while x < k:            
            
            teste = True
 
            df_aux = df.filter(F.col("distance_oq") > 0).orderBy("distance_oq")
  
            first_row = df.first() # TOO MUCH EXPENSIVE
            
            # dist_oq
            dist_oq = first_row.distance_oq
            # coord_oi
            coord = first_row.fv
            # index_row
            indexR = first_row.index_row
            # index_row
            id_row = first_row.id
            
            # min lower bound
            minimo = df.filter(df["index_row"] != first_row.index_row)/
                       .agg({f"lower_bound_oq_{int(oq.idOq)}": "min"})/
                       .collect()[0][0]  ## TOO MUCH EXPENSIVE
            
            for val in influences:
                
                aux.append([id_row, val[1], distance.euclidean(coord, val[0])])
                if (val[1] > distance.euclidean(coord, val[0])):
                    df = df.filter(df.index_row  != indexR)
                    coordenadas.append(coord)    
                    teste = False  
            
            if teste == False:
                continue
                  
            if (dist_oq <= minimo):               
                
                inputValues = (coord, dist_oq)
                influences.append([coord, dist_oq])
                influence_dictio[f"{id_row}"] = inputValues

                df = df.filter(df.index_row  != indexR)
                coordenadas.append(coord)
                x += 1
                
            if (dist_oq > minimo):
                
                # get everybody whos lower bound is less than dist(oi,oq)
                df = df.withColumn("distance_oq",\
                                   when((F.col(f"lower_bound_oq_{int(oq.idOq)}") <= dist_oq),\
                                   simpleF(oq.fvOq)(F.col('fv'))).\
                                   otherwise(F.col("distance_oq")))

                 
                get_min = df.filter((df[f"lower_bound_oq_{int(oq.idOq)}"] <= dist_oq)).\
                                    orderBy("distance_oq", ascending=True).\
                                    first() ## TOO MUCH EXPENSIVE
                
             
                # dist_oq
                dist_oq = get_min.distance_oq
                # coord_oi
                coord = get_min.fv
                # index_row
                indexR = get_min.index_row
                # index_row
                id_row = get_min.id  
                
                for val in influences:
                    
                    aux.append([id_row, val[1], distance.euclidean(coord, val[0])])
                    if (val[1] > distance.euclidean(coord, val[0])):
                        teste = False
                        
                        df = df.filter(df.index_row  != indexR)
                        coordenadas.append(coord)
                        teste = False
                
                if teste == False:
                    continue
                
                inputValues = (coord, dist_oq)
                influences.append([coord, dist_oq])
                influence_dictio[f"{id_row}"] = inputValues

                df = df.filter(df.index_row  != indexR)
                coordenadas.append(coord)
                x += 1
                
                
    dropped = df.filter(df.distance_oq.isNull()).count()
    
    return(coordenadas, influence_dictio, dropped)
   
