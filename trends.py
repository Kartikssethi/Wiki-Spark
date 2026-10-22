import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from common import get_spark, load_clickstream, MONTHS

MIN_PEAK = 200_000      # only track pages that reach at least this many views in some month
MIN_PREV = 1_000        # floor for last month's views in growth ratios (avoids dividing by ~0)
BREAKOUT_MIN = 500_000  # a "breakout" month must have at least this many views

os.makedirs("results", exist_ok=True)


def save(sdf, name):
    """Collect a SMALL Spark result to pandas, print it, and save results/<name>.csv"""
    pdf = sdf.toPandas()
    pdf.to_csv(f"results/{name}.csv", index=False)
    print(f"\n== {name} ==")
    print(pdf.to_string(index=False))


spark = get_spark("WikiTrends")
df = load_clickstream(spark)

# 1. Views per page per month
monthly = df.groupBy(F.col("curr").alias("page"), "month").agg(F.sum("n").alias("views"))

# 2. Keep pages big enough to matter, and give each one a row for EVERY month
#    (a page missing from a month had 0 recorded views; without this, lag() would
#    compare e.g. March with January and miss the jump)
big_pages = monthly.groupBy("page").agg(F.max("views").alias("peak")) \
    .filter(F.col("peak") >= MIN_PEAK).select("page")
all_months = spark.createDataFrame([(f"2026-{m}",) for m in MONTHS], ["month"])
grid = big_pages.crossJoin(all_months) \
    .join(monthly, ["page", "month"], "left") \
    .fillna({"views": 0}) \
    .cache()
print("Pages tracked:", big_pages.count())

# 3. Compare every month with the month before it (window function: lag over each page's timeline)
by_page = Window.partitionBy("page").orderBy("month")
changes = grid.withColumn("prev_views", F.lag("views").over(by_page)) \
    .filter(F.col("prev_views").isNotNull()) \
    .withColumn("jump", F.col("views") - F.col("prev_views")) \
    .withColumn("growth_x", F.round(F.col("views") / F.greatest(F.col("prev_views"), F.lit(MIN_PREV)), 1)) \
    .cache()

cols = ["page", "month", "prev_views", "views", "jump", "growth_x"]


def best_month_per_page(sdf, order_col):
    """Keep only each page's single most extreme month, so one page can't fill the whole list"""
    w = Window.partitionBy("page").orderBy(order_col)
    return sdf.withColumn("r", F.row_number().over(w)).filter(F.col("r") == 1).drop("r")


# 4. Biggest spikes: largest month-over-month increase in views
save(best_month_per_page(changes, F.desc("jump"))
     .orderBy(F.desc("jump")).select(cols).limit(25),
     "trend_biggest_spikes")

# 5. Event timeline: the top 3 risers of each month (January has no previous month)
by_month = Window.partitionBy("month").orderBy(F.desc("jump"))
save(changes.withColumn("rank", F.row_number().over(by_month))
            .filter(F.col("rank") <= 3)
            .orderBy("month", "rank").select("month", "rank", "page", "prev_views", "views", "jump"),
     "trend_monthly_timeline")

# 6. Breakouts: pages that came from almost nothing (highest growth multiple)
save(best_month_per_page(changes.filter(F.col("views") >= BREAKOUT_MIN), F.desc("growth_x"))
     .orderBy(F.desc("growth_x")).select(cols).limit(25),
     "trend_breakouts")

# 7. Biggest drops: pages whose attention fell fastest
save(best_month_per_page(changes, F.asc("jump"))
     .orderBy(F.asc("jump")).select(cols).limit(25),
     "trend_biggest_drops")

# 8. One-month wonders vs evergreen pages
shape = grid.groupBy("page").agg(F.sum("views").alias("total_views"),
                                 F.max("views").alias("peak_views"),
                                 F.min("views").alias("lowest_month")) \
    .withColumn("peak_share", F.round(F.col("peak_views") / F.col("total_views"), 2)) \
    .withColumn("stability", F.round(F.col("lowest_month") / F.col("peak_views"), 2))
# peak_share: fraction of all 9 months' views that came in the single best month (1.0 = all in one month)
# stability:  worst month / best month (1.0 = perfectly flat)

save(shape.filter(F.col("peak_share") >= 0.7).orderBy(F.desc("peak_views")).limit(20),
     "trend_one_month_wonders")
save(shape.filter(F.col("stability") >= 0.8).orderBy(F.desc("total_views")).limit(20),
     "trend_evergreen")

spark.stop()
