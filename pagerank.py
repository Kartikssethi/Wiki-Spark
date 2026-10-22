import os
from pyspark.sql import functions as F
from common import get_spark, load_clickstream

DAMPING = 0.85     # chance a reader keeps clicking links instead of jumping somewhere random
ITERATIONS = 3     # quick test run; set to 10 for the final results

os.makedirs("results", exist_ok=True)
spark = get_spark("WikiPageRank")
df = load_clickstream(spark)

# 1. Build the graph: article -> article edges, weighted by total link clicks over 9 months
edges_named = df.filter(F.col("type") == "link") \
    .groupBy("prev", "curr").agg(F.sum("n").alias("w"))

# 2. Give every page a numeric id (joins on numbers are much cheaper than on long title strings)
pages = edges_named.select(F.col("prev").alias("title")) \
    .union(edges_named.select(F.col("curr").alias("title"))) \
    .distinct() \
    .withColumn("id", F.monotonically_increasing_id()) \
    .localCheckpoint()
N = pages.count()

src = pages.select(F.col("title").alias("prev"), F.col("id").alias("src"))
dst = pages.select(F.col("title").alias("curr"), F.col("id").alias("dst"))
edges = edges_named.join(src, "prev").join(dst, "curr").select("src", "dst", "w")

# 3. Turn click counts into probabilities: share of a page's outgoing clicks that go to each target
out_w = edges.groupBy("src").agg(F.sum("w").alias("out_w"))
edges = edges.join(out_w, "src") \
    .select("src", "dst", (F.col("w") / F.col("out_w")).alias("p")) \
    .localCheckpoint()
has_out = edges.select(F.col("src").alias("id")).distinct().localCheckpoint()
print(f"Graph: {N:,} pages, {edges.count():,} edges")

# 4. Iterate: every page passes its rank to the pages it links to, in proportion to p
ranks = pages.select("id", F.lit(1.0 / N).alias("rank"))
for i in range(ITERATIONS):
    # rank sitting on pages with no outgoing links is spread evenly over all pages
    dangling = ranks.join(has_out, "id", "left_anti").agg(F.sum("rank")).first()[0] or 0.0

    contribs = edges.join(ranks.withColumnRenamed("id", "src"), "src") \
        .groupBy("dst").agg(F.sum(F.col("rank") * F.col("p")).alias("c")) \
        .withColumnRenamed("dst", "id")

    new_ranks = pages.select("id").join(contribs, "id", "left") \
        .select("id", ((1 - DAMPING) / N
                       + DAMPING * (F.coalesce(F.col("c"), F.lit(0.0)) + dangling / N)).alias("rank")) \
        .localCheckpoint()

    delta = new_ranks.join(ranks.withColumnRenamed("rank", "old"), "id") \
        .agg(F.sum(F.abs(F.col("rank") - F.col("old")))).first()[0]
    print(f"Iteration {i + 1}: total change = {delta:.6f}")
    ranks = new_ranks

# 5. Attach titles and total views, then compare the two rankings
views = df.groupBy(F.col("curr").alias("title")).agg(F.sum("n").alias("total_views"))
pr_named = ranks.join(pages, "id").select("title", F.col("rank").alias("pagerank"))
result = views.join(pr_named, "title", "outer").fillna({"total_views": 0})

top_pr = result.orderBy(F.desc_nulls_last("pagerank")).limit(5000).toPandas()
top_views = result.orderBy(F.desc_nulls_last("total_views")).limit(5000).toPandas()
top_pr["pr_position"] = range(1, len(top_pr) + 1)
top_views["views_position"] = range(1, len(top_views) + 1)
top_pr = top_pr.merge(top_views[["title", "views_position"]], on="title", how="left")
top_views = top_views.merge(top_pr[["title", "pr_position"]], on="title", how="left")

top_pr.to_csv("results/pagerank_top.csv", index=False)
top_views.to_csv("results/views_top.csv", index=False)
print("\nTop 15 by PageRank (views_position blank = not in top 5000 by views):")
print(top_pr.head(15).to_string(index=False))

spark.stop()
