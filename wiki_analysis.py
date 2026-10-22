from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("WikiClickstream2026") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.host", "localhost") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

# Hide the per-block "Not enough space to cache ... persisting to disk" WARN spam;
# spilling the cache to disk is expected with ~14GB of input and 8GB of driver memory
spark.sparkContext.setLogLevel("ERROR")

schema = StructType([
    StructField("prev", StringType(), True),
    StructField("curr", StringType(), True),
    StructField("type", StringType(), True),
    StructField("n", LongType(), True),
])

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]

dfs = []
for m in months:
    filename = f"Data/clickstream-enwiki-2026-{m}.tsv"
    # quote="" disables quote handling: titles like "Weird_Al"_Yankovic start with a literal "
    temp = spark.read.csv(filename, sep="\t", schema=schema, quote="") \
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

# Top traffic sources for one page (change TARGET_PAGE to explore others).
# prev values starting with "other-" are aggregate sources (other-search, other-external,
# other-empty, other-internal, other-other), the rest are Wikipedia articles
TARGET_PAGE = "ChatGPT"

target = df.filter(F.col("curr") == TARGET_PAGE)

print(f"Top 20 traffic sources for {TARGET_PAGE}:")
target.groupBy("prev", "type").agg(F.sum("n").alias("clicks")) \
  .orderBy(F.desc("clicks")).show(20, truncate=False)

print(f"Monthly views of {TARGET_PAGE}:")
target.groupBy("month").agg(F.sum("n").alias("views")) \
  .orderBy("month").show()

# Strongest article -> article navigation edges (internal links only), summed over all months
links = df.filter(F.col("type") == "link")

print("Top 20 strongest navigation edges:")
links.groupBy("prev", "curr").agg(F.sum("n").alias("clicks")) \
  .orderBy(F.desc("clicks")).show(20, truncate=False)

# Pages that send the most readers onward via internal links
print("Top 15 pages by outgoing link clicks:")
links.groupBy("prev").agg(F.sum("n").alias("outgoing_clicks"),
                          F.countDistinct("curr").alias("distinct_targets")) \
  .orderBy(F.desc("outgoing_clicks")).show(15, truncate=False)

spark.stop()
