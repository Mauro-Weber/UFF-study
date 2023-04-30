
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:55:45 2023

@author: weber
"""

def reduce_knn(pq, count, k):

    knn_list = sum(pq[::2], [])
    knn_list = [[value[1],-value[0]] for value in knn_list]
    sorted_list = sorted(knn_list, key=lambda x: x[1])    
    top_k_elements = sorted_list[:k]
    h_list = pq[1::2]
    drops = count - sum(h_list)

    return top_k_elements, drops