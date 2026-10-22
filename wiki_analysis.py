from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("WikiClickstream2026") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

schema = StructType([
    StructField("prev", StringType(), True),
    StructField("curr", StringType(), True),
    StructField("type", StringType(), True),
    StructField("n", IntegerType(), True),
])

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]

dfs = []
for m in months:
    filename = f"Data/clickstream-enwiki-2026-{m}.tsv"
    temp = spark.read.csv(filename, sep="\t", schema=schema) \
        .withColumn("month", F.lit(f"2026-{m}"))
    dfs.append(temp)

df = dfs[0]
for other in dfs[1:]:
    df = df.union(other)

df = df.na.drop(subset=["prev", "curr"])
df.cache()

print("Total rows across all 9 months:", df.count())

# Monthly total traffic
df.groupBy("month").agg(F.sum("n").alias("total_clicks")) \
  .orderBy("month").show()

# Top 15 most-visited pages overall (across all months)
df.groupBy("curr").agg(F.sum("n").alias("total_views")) \
  .orderBy(F.desc("total_views")).show(15, truncate=False)

# Referral type breakdown
df.groupBy("type").agg(F.sum("n").alias("total")).show()

spark.stop()