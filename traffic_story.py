import os
from pyspark.sql import functions as F
from common import get_spark, load_clickstream

CASE_PAGE = "Artificial_intelligence"   # case-study page; any title works
MIN_VIEWS = 1_000_000                   # only rank pages with at least this many views

os.makedirs("results", exist_ok=True)


def save(sdf, name):
    """Collect a SMALL Spark result to pandas, print it, and save results/<name>.csv"""
    pdf = sdf.toPandas()
    pdf.to_csv(f"results/{name}.csv", index=False)
    print(f"\n== {name} ==")
    print(pdf.to_string(index=False))


spark = get_spark("WikiTrafficStory")
df = load_clickstream(spark).cache()

# ---------------------------------------------------------------------------
# Part 1: The entry door - how do readers arrive on a page?
# ---------------------------------------------------------------------------
source = F.when(F.col("type") == "link", "Internal link") \
    .when(F.col("type") == "other", "Article, no direct link") \
    .when(F.col("prev") == "other-search", "Search engine") \
    .when(F.col("prev") == "other-empty", "Direct / no referrer") \
    .when(F.col("prev") == "other-external", "Other website") \
    .when(F.col("prev") == "other-internal", "Other Wikimedia site") \
    .otherwise("Other")

by_source = df.groupBy(source.alias("source")).agg(F.sum("n").alias("clicks"))
total = by_source.agg(F.sum("clicks")).first()[0]
save(by_source.withColumn("share_pct", F.round(F.col("clicks") / total * 100, 2))
              .orderBy(F.desc("clicks")),
     "story_sources")

# Same split per month, to see if the mix is stable over time
monthly = df.groupBy("month", source.alias("source")).agg(F.sum("n").alias("clicks"))
month_total = monthly.groupBy("month").agg(F.sum("clicks").alias("month_total"))
save(monthly.join(month_total, "month")
            .withColumn("share_pct", F.round(F.col("clicks") / F.col("month_total") * 100, 2))
            .select("month", "source", "clicks", "share_pct")
            .orderBy("month", F.desc("clicks")),
     "story_sources_monthly")

# ---------------------------------------------------------------------------
# Part 2: The rabbit hole - which pages are reached by browsing, and which
# pages push readers further in? (Clickstream has no sessions, so this is
# measured per page rather than as "clicks deep" per reader.)
# ---------------------------------------------------------------------------
arrivals = df.groupBy(F.col("curr").alias("page")).agg(
    F.sum("n").alias("views"),
    F.sum(F.when(F.col("prev") == "other-search", F.col("n")).otherwise(0)).alias("from_search"),
    F.sum(F.when(F.col("type") == "link", F.col("n")).otherwise(0)).alias("from_links"),
)
onward = df.filter(F.col("type") == "link") \
    .groupBy(F.col("prev").alias("page")).agg(F.sum("n").alias("onward_clicks"))

pages = arrivals.join(onward, "page", "left").fillna({"onward_clicks": 0}) \
    .filter(F.col("views") >= MIN_VIEWS) \
    .withColumn("search_pct", F.round(F.col("from_search") / F.col("views") * 100, 1)) \
    .withColumn("links_pct", F.round(F.col("from_links") / F.col("views") * 100, 1)) \
    .withColumn("onward_per_view", F.round(F.col("onward_clicks") / F.col("views"), 3)) \
    .cache()

cols = ["page", "views", "search_pct", "links_pct", "onward_per_view"]

# Front doors: pages people land on straight from Google
save(pages.orderBy(F.desc("search_pct")).select(cols).limit(20), "story_search_doors")

# Rabbit-hole pages: pages people mostly reach by clicking around Wikipedia
save(pages.orderBy(F.desc("links_pct")).select(cols).limit(20), "story_rabbit_hole_pages")

# Gateways: pages that send the most readers onward per view
save(pages.orderBy(F.desc("onward_per_view")).select(cols).limit(20), "story_gateways")

# ---------------------------------------------------------------------------
# Part 3: Case study - where do readers of CASE_PAGE come from and go next?
# ---------------------------------------------------------------------------
save(pages.filter(F.col("page") == CASE_PAGE).select(cols), "story_case_summary")

save(df.filter(F.col("curr") == CASE_PAGE)
       .groupBy("prev", "type").agg(F.sum("n").alias("clicks"))
       .orderBy(F.desc("clicks")).limit(15),
     "story_case_inbound")

save(df.filter((F.col("prev") == CASE_PAGE) & (F.col("type") == "link"))
       .groupBy("curr").agg(F.sum("n").alias("clicks"))
       .orderBy(F.desc("clicks")).limit(15),
     "story_case_outbound")

spark.stop()
