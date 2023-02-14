# calculate the euclidian distance from two points
import math


def euclidean_distance(p1, p2):
    return math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))


def kNearestNeighbor(dataframe, point, k):
    
    results = []
    data_collect = dataframe.collect()
    
    # create column with distance from a given point 
    for row in data_collect:
        x = float(row["x"])
        y = float(row["y"])
        partition_point = (x,y)
        result = euclidian_distance(point,partition_point)
        results.append(result)
    
    fourth_dataframe = spark.createDataFrame([(l,) for l in results], ['distancefrompoint'])
    dataframe_withindex = dataframe.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    fourth_dataframe_withindex = fourth_dataframe.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    
    # join the two dataframes (poitns, distance from point)
    final_dataframe = dataframe_withindex.join(fourth_dataframe_withindex, dataframe_withindex.row_idx == fourth_dataframe_withindex.row_idx)
    final_df = final_dataframe.sort(F.col("distancefrompoint").asc()).drop("row_idx")
    
    
    # list the k nearest points
    newfinal_df = final_df.limit(k)
    newfinal_df.show()
    final_lista = getListOfPoints(newfinal_df)
    
    #resultdataframe.show()
    return(final_lista)