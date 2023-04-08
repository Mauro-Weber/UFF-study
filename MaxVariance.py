class MaxVariance:
    
    def __init__(self, df, nPivots):
        self.df = df
        self.nPivots = nPivots
        
    def distance_udf(self):
        return F.udf(lambda x, y: float(distance.euclidean(x, y)), DoubleType())
    
    def calculate_sum_of_squares(self, df, size):
        return df.groupBy("features", "avg(distance)").agg(
            (sum(pow(col("distance") - col("avg(distance)"), 2))/size).alias("variance")
        )
    
    def find_max_variance(self):
        # create sample dataset
        dataset = self.df.select("id","coord_x", "coord_y")

        # create a random sample of the input dataset
        sample1 = dataset.sample(False, 0.1, seed=42)
        sample2 = dataset.sample(False, 0.1, seed=24)

        # convert dataset to vectors
        cNames = dataset.columns
        cNames.remove("id")
        assembler = VectorAssembler(inputCols=cNames,outputCol="features")
        
        
        #assembler = VectorAssembler(inputCols=dataset.columns, outputCol="features")
        dataset_vectors = assembler.transform(dataset).select("features")
        sample_vectors1 = assembler.transform(sample1).select("features")
        sample_vectors2 = assembler.transform(sample2).select("features").withColumnRenamed("features","features2")

        # calculate mean distance between objects in part1 and part2
        cross_df = sample_vectors1.crossJoin(sample_vectors2)
        distance = cross_df.withColumn("distance", self.distance_udf()("features", "features2"))
        mean_distance = distance.groupBy("features").agg({"distance": "mean"})
        
        # calculate variance between objects in part1 and part2
        new_df = mean_distance.join(distance, on="features", how="inner")
        variance_df = self.calculate_sum_of_squares(new_df, (sample_vectors2.count() + 1))
        
        # save the variances and sort it
        rows = variance_df.select("features", "variance").collect()
        list_var = [(row.features, row.variance) for row in rows]
        sorted_lst = sorted(list_var, key=lambda x: x[1], reverse=True)[0:self.nPivots]
        
        return sorted_lst
