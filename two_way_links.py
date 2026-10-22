import os
from pyspark.sql import functions as F
from common import get_spark, load_clickstream

MIN_CLICKS = 10_000        # a direction counts as "strong" with at least this many clicks over 9 months
SEED_PAGE = "Tom_Holland"  # page whose two-way neighbourhood we print as an example cluster

os.makedirs("results", exist_ok=True)


def save(sdf, name):
    """Collect a SMALL Spark result to pandas, print it, and save results/<name>.csv"""
    pdf = sdf.toPandas()
    pdf.to_csv(f"results/{name}.csv", index=False)
    print(f"\n== {name} ==")
    print(pdf.to_string(index=False))


spark = get_spark("WikiTwoWayLinks")
df = load_clickstream(spark)

# 1. Article -> article link clicks, summed over 9 months; keep only strong directions
#    (filtering before the self-join shrinks it from tens of millions of edges to a small set)
strong = df.filter(F.col("type") == "link") \
    .groupBy("prev", "curr").agg(F.sum("n").alias("clicks")) \
    .filter(F.col("clicks") >= MIN_CLICKS) \
    .cache()

# 2. Self-join: match every A -> B edge with its reverse B -> A edge
forward = strong.select(F.col("prev").alias("page_a"), F.col("curr").alias("page_b"),
                        F.col("clicks").alias("a_to_b"))
reverse = strong.select(F.col("curr").alias("page_a"), F.col("prev").alias("page_b"),
                        F.col("clicks").alias("b_to_a"))

pairs = forward.join(reverse, ["page_a", "page_b"]) \
    .filter(F.col("page_a") < F.col("page_b")) \
    .withColumn("total", F.col("a_to_b") + F.col("b_to_a")) \
    .withColumn("balance", F.round(F.least("a_to_b", "b_to_a") / F.greatest("a_to_b", "b_to_a"), 2)) \
    .cache()
# page_a < page_b keeps each pair once (A<->B, not also B<->A).
# balance: 1.0 = equal traffic both ways, 0.1 = one direction is 10x stronger.

n_strong = strong.count()
n_pairs = pairs.count()
print(f"\nStrong one-way edges (>= {MIN_CLICKS:,} clicks): {n_strong:,}")
print(f"Two-way pairs (strong in both directions): {n_pairs:,}")
print(f"Share of strong edges that are reciprocated: {2 * n_pairs / n_strong * 100:.1f}%")

# 3. Strongest two-way pairs overall
save(pairs.orderBy(F.desc("total")).limit(25), "twoway_top_pairs")

# 4. Truly mutual pairs: nearly equal traffic in both directions
save(pairs.filter(F.col("balance") >= 0.8).orderBy(F.desc("total")).limit(25),
     "twoway_balanced_pairs")

# 5. Lopsided pairs: two-way, but one side pulls far harder than the other
save(pairs.filter(F.col("balance") <= 0.2).orderBy(F.desc("total")).limit(25),
     "twoway_lopsided_pairs")

# 6. Cluster hubs: pages with the most two-way partners
both_sides = pairs.select(F.col("page_a").alias("page"), "total") \
    .union(pairs.select(F.col("page_b").alias("page"), "total"))
save(both_sides.groupBy("page").agg(F.count("*").alias("two_way_partners"),
                                    F.sum("total").alias("two_way_clicks"))
               .orderBy(F.desc("two_way_partners")).limit(25),
     "twoway_cluster_hubs")

# 7. Example cluster: everything SEED_PAGE is mutually linked with
save(pairs.filter((F.col("page_a") == SEED_PAGE) | (F.col("page_b") == SEED_PAGE))
          .select(F.when(F.col("page_a") == SEED_PAGE, F.col("page_b"))
                   .otherwise(F.col("page_a")).alias("partner"),
                  F.when(F.col("page_a") == SEED_PAGE, F.col("a_to_b"))
                   .otherwise(F.col("b_to_a")).alias("from_seed"),
                  F.when(F.col("page_a") == SEED_PAGE, F.col("b_to_a"))
                   .otherwise(F.col("a_to_b")).alias("to_seed"),
                  "total", "balance")
          .orderBy(F.desc("total")),
     "twoway_seed_cluster")

spark.stop()
