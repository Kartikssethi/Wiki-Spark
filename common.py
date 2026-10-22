from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType
from pyspark.sql import functions as F

MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]

SCHEMA = StructType([
    StructField("prev", StringType(), True),
    StructField("curr", StringType(), True),
    StructField("type", StringType(), True),
    StructField("n", LongType(), True),
])


def get_spark(app_name):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    # Hide the per-block "Not enough space to cache ... persisting to disk" WARN spam
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_clickstream(spark):
    dfs = []
    for m in MONTHS:
        # quote="" disables quote handling: titles like "Weird_Al"_Yankovic start with a literal "
        temp = spark.read.csv(f"Data/clickstream-enwiki-2026-{m}.tsv", sep="\t",
                              schema=SCHEMA, quote="") \
            .withColumn("month", F.lit(f"2026-{m}"))
        dfs.append(temp)

    df = dfs[0]
    for other in dfs[1:]:
        df = df.union(other)
    return df.na.drop(subset=["prev", "curr"])
