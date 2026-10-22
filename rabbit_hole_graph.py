import os
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from common import get_spark, load_clickstream

SEEDS = ["Artificial_intelligence", "Tom_Holland", "2026_FIFA_World_Cup"]  # starting pages
FAN_OUT = [4, 3, 2]   # how many next pages to follow at hop 1, hop 2, hop 3
PATH_STEPS = 8        # length of the "most likely path"
MIN_PATH_CLICKS = 1_000  # stop a path once the next step has fewer clicks than this

os.makedirs("results", exist_ok=True)

spark = get_spark("WikiRabbitHoleGraph")
df = load_clickstream(spark)

# Article -> article link clicks over 9 months, plus each page's share of its own onward clicks
links = df.filter(F.col("type") == "link") \
    .groupBy("prev", "curr").agg(F.sum("n").alias("clicks"))
onward_total = links.groupBy("prev").agg(F.sum("clicks").alias("onward_total"))
links = links.join(onward_total, "prev") \
    .withColumn("share_pct", F.round(F.col("clicks") / F.col("onward_total") * 100, 1)) \
    .drop("onward_total") \
    .cache()
print("Link edges:", links.count())


def top_next(pages, k):
    """For each page in `pages`, its k most-clicked next pages"""
    w = Window.partitionBy("prev").orderBy(F.desc("clicks"))
    return links.filter(F.col("prev").isin(pages)) \
        .withColumn("r", F.row_number().over(w)) \
        .filter(F.col("r") <= k).drop("r") \
        .toPandas()


# 1. Rabbit-hole tree: follow the top next pages hop by hop from each seed
tree_rows = []
for seed in SEEDS:
    frontier, seen = [seed], {seed}
    for hop, k in enumerate(FAN_OUT, start=1):
        step = top_next(frontier, k)
        step.insert(0, "hop", hop)
        step.insert(0, "seed", seed)
        tree_rows.append(step)
        # only expand pages we haven't visited yet (stops loops like A -> B -> A)
        frontier = [p for p in step["curr"].unique() if p not in seen]
        seen.update(frontier)
        print(f"{seed}: hop {hop} -> {len(step)} edges")
        if not frontier:
            break

tree = pd.concat(tree_rows, ignore_index=True)
tree = tree[["seed", "hop", "prev", "curr", "clicks", "share_pct"]]
tree.to_csv("results/rabbit_tree.csv", index=False)

# 2. Most likely path: always follow the most-clicked link to a page not yet on the path
path_rows = []
for seed in SEEDS:
    page, visited = seed, {seed}
    for step_no in range(1, PATH_STEPS + 1):
        options = top_next([page], 10)
        options = options[~options["curr"].isin(visited)]
        if options.empty or options.iloc[0]["clicks"] < MIN_PATH_CLICKS:
            break
        best = options.iloc[0]
        path_rows.append({"seed": seed, "step": step_no, "prev": page, "curr": best["curr"],
                          "clicks": best["clicks"], "share_pct": best["share_pct"]})
        page = best["curr"]
        visited.add(page)
    print(f"{seed}: path of {len(visited) - 1} steps")

paths = pd.DataFrame(path_rows)
paths.to_csv("results/rabbit_paths.csv", index=False)

print("\n== Most likely paths ==")
for seed, group in paths.groupby("seed", sort=False):
    print(" -> ".join([seed] + [f"{c} ({s}%)" for c, s in zip(group["curr"], group["share_pct"])]))

spark.stop()
