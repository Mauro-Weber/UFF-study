##
## GIVEN DATA
##

k = 10 #NEIGHBOR
pivots_list = [(i+1,[np.random.rand(1)[0],np.random.rand(1)[0]]) for i in range(pivots)] #PIVOTS


##
## Defines query objects dataframe 
##

oq = [(1.0,[0.0,0.0]),(2.0,[1.0,1.0]),(3.0,[0.5,0.5])] # fixed query objects for test 
oqColumns = ["idOq","fvOq"]
oqDF = spark.createDataFrame(data=oq, schema = oqColumns)

##
## Loading Dataframe
##

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("coord_x", FloatType(), True),
    StructField("coord_y", FloatType(), True)])

df = spark.read.csv("/home/weber/Documents/coordDF100.csv",schema=schema)        
size = df.count()
##SMALL SANITY CHECK - @PRODUCTION TESTAR LOADING
df = df.na.drop()

## define the struct for the dimensional feature vector
cNames = df.columns
cNames.remove("id")
assembler = VectorAssembler(
    inputCols=cNames,
    outputCol="fv")

## appends the fv into the dataframe as column
df = assembler.transform(df)


##
## Tranforming dataframe with distance calculus from pivots and oq
##


rows_oqDF = oqDF.collect()

## calculates the distance from oi to pivots
for i in range(len(pivots_list)):
    df = df.withColumn(f'distances_oi_pivot{i+1}', simpleF(pivots_list[i][1])(F.col('fv'))) 


##
## DISTANCE FROM OQ TO PIVOT FOR ALL OQs IN oqDF
##
for oq in rows_oqDF: 
    lista_columns = []
    for i in range(len(pivots_list)):
        df = df.withColumn(f"lower_oq_{int(oq.idOq)}_pivot_{i+1}", \
            abs (distance.euclidean(oq.fvOq, pivots_list[i][1]) - \
            F.col(f'distances_oi_pivot{i+1}')))

        lista_columns.append(f"lower_oq_{int(oq.idOq)}_pivot_{i+1}")

    df = df.withColumn(f"lower_bound_oq_{int(oq.idOq)}", greatest(*[col_name for col_name in lista_columns]))


