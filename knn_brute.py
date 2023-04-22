def knn(dataframe, oq, k):       
    
    ##CALCULATES THE DIST FROM ELEMENTS WITHIN THE DF TO THE QUERY POINT
    df = dataframe.withColumn('distances', simpleF(oq)(F.col('fv')))
    
    ##SORTS AND RETRIEVES THE TOP-K RESULTS
    resultSet = df.select("id", "distances").orderBy('distances').limit(k)
    resultSet.show()
    return(resultSet)

