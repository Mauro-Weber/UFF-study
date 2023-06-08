from scipy.spatial import distance
import heapq

def bridk_reduce(lst, k):
    influence_lst = []

    dropped_rows = lst[1::2]
    total_dropped = sum(dropped_rows)

    nearest_neighbors = sum(lst[::2], [])
    nearest_neighbors = [[elem[1],elem[0],elem[2]] for elem in nearest_neighbors]
    heapq.heapify(nearest_neighbors)

    influence_lst.append(nearest_neighbors[0])
    heapq.heappop(nearest_neighbors)

    size = len(nearest_neighbors)
    j = 0
    while (len(influence_lst) < k) and (j < size):
        j+=1
        infl_test = True 
        for val in influence_lst:
            if (val[0] > distance.euclidean(val[2], nearest_neighbors[0][2])):
                infl_test = False 

        if infl_test == False:
            heapq.heappop(nearest_neighbors) 

        else:
            influence_lst.append([nearest_neighbors[0][0],\
                                    nearest_neighbors[0][1],\
                                    nearest_neighbors[0][2]])
            heapq.heappop(nearest_neighbors)    

    influence_lst = [[value[1],value[0]] for value in influence_lst]

    return(influence_lst,total_dropped)