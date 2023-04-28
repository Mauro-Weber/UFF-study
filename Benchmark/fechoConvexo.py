#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:04:55 2023

@author: weber
"""

from pyspark.sql.types import DoubleType
from scipy.spatial import distance
import pyspark.sql.functions as F
import numpy as np


def distanceF(oq):
    return F.udf(lambda x: float(distance.euclidean(x, oq)), DoubleType())

class FechoConv:
    
    def __init__(self, df, nPivots):
        self.df = df["id","fv"]
        self.nPivots = nPivots
        self.foci_lst = []
        self.pivots = None
        self.edge = None    
    
    def return_pivots(self):
        focus_taken = []
        df_fc = self.df["id","fv"] 
        object_rand = df_fc.select("fv").rdd.takeSample(False, 1)[0][0]
        
        df_fc = df_fc.withColumn("distance_to_randObject", distanceF(object_rand)(F.col("fv")))
        farest_obj = df_fc.orderBy(F.col("distance_to_randObject").desc()).first()
        self.foci_lst.append([farest_obj.id, farest_obj.fv, farest_obj.distance_to_randObject])

        focus_taken.append(farest_obj.id)

        df_fc = df_fc.withColumn("distance_to_f1", distanceF(farest_obj.fv)(F.col("fv")))
        farest_obj_f1 = df_fc.orderBy(F.col("distance_to_f1").desc()).first()
        self.foci_lst.append([farest_obj_f1.id, farest_obj_f1.fv, farest_obj_f1.distance_to_randObject])
        
        focus_taken.append(farest_obj_f1.id)
        
        edge = farest_obj_f1.distance_to_randObject
        self.edge = edge
        
        df_fc = self.df["id","fv"]
        rows = df_fc.collect()
        i = 0
        
        while i<(self.nPivots-2):
            new_focus = []
            for row in rows:
                teste = True
 
                if row.id in focus_taken:
                    teste = False 
                
                if teste == True:
                    erro = 0
                    for elem in self.foci_lst:    
                        erro += np.absolute(self.edge - distance.euclidean(elem[1],row.fv))
                    new_focus.append([row.id, row.fv, erro])
            minor_obj = max(new_focus, key=lambda x: x[2])
            i+=1
            focus_taken.append(minor_obj[0])
            self.foci_lst.append(minor_obj)
        
        pivots = []
        for l in self.foci_lst:
            pivots.append(l[1])
        self.pivots = pivots
        sorted_pivots = [(i, tupla) for i, tupla in enumerate(self.pivots, 1)]
        
        return sorted_pivots
