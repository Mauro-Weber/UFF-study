#!/usr/bin/env python3

# Import SparkSession
from pyspark.sql import SparkSession

from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import *


distance_udf = F.udf(lambda x,y: float(distance.euclidean(x, y)), DoubleType())

def simpleF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())




def bridk_simple(df, oq, k):
      
    
    ##CALCULATES THE DIST FROM ELEMENTS WITHIN THE DF TO THE QUERY POINT
    df = df.withColumn('distances', simpleF(oq)(F.col('fv'))).orderBy('distances')    
    
    # auxiliary variables
    list_result = []
    h = 1
    dropped = 0
    first_run = True
    
    coordenadas = []
    dic_ent = {}
    aux = []
    influences = []
    
    
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
            dic_ent[f"{row.id}"] = [row["fv"],row["distances"]]
            first_run =  False
            continue
        
        
        teste = True
        coordenadas.append(row["fv"])
        for val in influences:
            aux.append([row["id"], val[1], distance.euclidean(row["fv"], val[0])])
            if (val[1] > distance.euclidean(row["fv"], val[0])):
                teste = False

        if teste == False:
            continue
        
        
        
            
        distanceToR = distance.euclidean(point_influence,row["fv"])
    
        if distanceToR > r_influence:

            #GET ELEMENT
            id_influence = row["id"]
            point_influence = row["fv"]
            r_influence = row["distances"]

            list_result.append((row["id"],row["fv"],row["distances"]))
            
            
            dic_ent[f"{row.id}"] = [row["fv"],row["distances"]]
            influences.append([row["fv"], row["distances"]])
            
            h += 1
        else:
            dropped += 1
        
        if h == k:
            break
    
    return (list_result,coordenadas,dropped,dic_ent) 