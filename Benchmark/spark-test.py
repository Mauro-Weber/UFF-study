# from pyspark.sql import SparkSession

# spark = SparkSession.builder \
#     .appName("SparkTest") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.task.cpus", "1") \
#     .getOrCreate()

# print("Número de cores configurados:", spark.sparkContext.defaultParallelism)

# data = range(100000)
# rdd = spark.sparkContext.parallelize(data)
# result = rdd.map(lambda x: x * x).collect()

# # Imprima o resultado da computação
# print("Resultado ENDED")

# spark.stop()



# from pyspark.sql import SparkSession

# spark = SparkSession.builder \
#     .appName("ResourceConfig") \
#     .getOrCreate()

# # Obter o número total de cores disponíveis
# total_cores = spark.sparkContext.getConf().get("spark.executor.instances") * \
#               spark.sparkContext.getConf().get("spark.executor.cores")

# print("Número total de cores disponíveis:", total_cores)

# # Obter o número de cores alocados por executor
# cores_per_executor = spark.sparkContext.getConf().get("spark.executor.cores")

# print("Número de cores alocados por executor:", cores_per_executor)

# spark.stop()


# from pyspark.sql import SparkSession

# spark = SparkSession.builder \
#     .appName("SparkTest") \
#     .getOrCreate()

# print("Número de cores disponíveis:", spark.sparkContext.defaultParallelism)

# spark.stop()

# from pyspark.sql import SparkSession

# spark = SparkSession.builder \
#     .appName("SparkTest") \
#     .getOrCreate()

# print("Número de cores disponíveis:", spark.sparkContext.defaultParallelism)

# spark.stop()


from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkTest") \
    .config("spark.executor.cores", "4") \
    .config("spark.task.cpus", "1") \
    .getOrCreate()

print("Número de cores configurados:", spark.conf.get("spark.executor.cores"))
print("Número de CPUs por tarefa configurados:", spark.conf.get("spark.task.cpus"))

data = range(100000)
rdd = spark.sparkContext.parallelize(data)
result = rdd.map(lambda x: x * x).collect()

 # Imprima o resultado da computação
print("Resultado ENDED")


print("Número de cores configurados:", spark.conf.get("spark.executor.cores"))
print("Número de CPUs por tarefa configurados:", spark.conf.get("spark.task.cpus"))

spark.stop()
