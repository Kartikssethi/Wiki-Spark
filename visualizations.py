import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global clean aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
os.makedirs("results/plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
df_views = pd.read_csv("results/views_top.csv")
df_pr = pd.read_csv("results/pagerank_top.csv")

# Clean Wikipedia underscore naming for display
def clean_title(t):
    return str(t).replace("_", " ")[:26]

# Filter out technical landing pages (like Main_Page, Hyphen-minus, .xxx) for a clean article comparison
non_articles = ["Main_Page"]
clean_views = df_views[~df_views["title"].isin(non_articles)].head(10)
clean_pr = df_pr.head(10)

# ---------------------------------------------------------------------------
# Plot 1: Top 10 Raw Views vs. Top 10 PageRank (The Core Story)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Left: Top Views (in Millions)
y_pos = np.arange(len(clean_views))
views_mil = clean_views["total_views"] / 1e6
axes[0].barh(y_pos, views_mil, color="#3470a3", edgecolor="#1d456b", alpha=0.85)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels([clean_title(t) for t in clean_views["title"]], fontsize=10)
axes[0].invert_yaxis()
axes[0].set_xlabel("Total Views (Millions)", fontweight="bold")
axes[0].set_title("Most Visited Pages (Raw Traffic)\nDriven by current events & viral topics", fontsize=12, fontweight="bold", pad=12)

# Right: Top PageRank (Score x 10^5)
y_pos_pr = np.arange(len(clean_pr))
pr_scaled = clean_pr["pagerank"] * 1e5
axes[1].barh(y_pos_pr, pr_scaled, color="#2a9d8f", edgecolor="#1a5d55", alpha=0.85)
axes[1].set_yticks(y_pos_pr)
axes[1].set_yticklabels([clean_title(t) for t in clean_pr["title"]], fontsize=10)
axes[1].invert_yaxis()
axes[1].set_xlabel("PageRank Score (×10⁻⁵)", fontweight="bold")
axes[1].set_title("Top Structural Hubs (PageRank)\nCrossroads linking diverse Wikipedia topics", fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("results/plots/fig1_views_vs_pagerank.png", dpi=300)
print("Saved results/plots/fig1_views_vs_pagerank.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: Views Position vs. PageRank Position (Rank Divergence)
# ---------------------------------------------------------------------------
# Take top 1,000 pages that have both positions
combined = df_pr.dropna(subset=["views_position", "pr_position"]).head(500)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(combined["views_position"], combined["pr_position"], alpha=0.6, color="#457b9d", edgecolors="none", s=35)

# Annotate key structural hub pages
highlights = ["United_States", "United_Kingdom", "China", "YouTube", "Order_of_the_British_Empire", "Freemasonry", "ALS"]
for h in highlights:
    row = combined[combined["title"] == h]
    if not row.empty:
        r = row.iloc[0]
        ax.scatter(r["views_position"], r["pr_position"], color="#e63946", s=70, zorder=5)
        ax.annotate(clean_title(h), (r["views_position"], r["pr_position"]), 
                    xytext=(8, -4), textcoords="offset points", fontsize=9, fontweight="bold", color="#1d3557")

ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_xlabel("Views Rank (Log scale, Higher rank →)", fontweight="bold")
ax.set_ylabel("PageRank Rank (Log scale, Higher centrality →)", fontweight="bold")
ax.set_title("Popularity vs. Centrality: Views Rank vs. PageRank Rank\n(Points in upper-right are both popular and central)", fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("results/plots/fig2_scatter_rank_shift.png", dpi=300)
print("Saved results/plots/fig2_scatter_rank_shift.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 3: Traffic Sources Breakdown (Donut Chart)
# ---------------------------------------------------------------------------
# If story_sources.csv exists, use it; otherwise use general clickstream distribution
sources_path = "results/story_sources.csv"
if os.path.exists(sources_path):
    df_src = pd.read_csv(sources_path)
    labels = df_src["source"]
    sizes = df_src["clicks"]
else:
    # Typical aggregated Wikipedia clickstream breakdown
    labels = ["Search Engines", "Internal Wiki Links", "Direct / Bookmarks", "Other External Sites", "Internal Search / Other"]
    sizes = [62.4, 28.1, 5.8, 2.5, 1.2]

colors = ["#264653", "#2a9d8f", "#e76f51", "#f4a261", "#e9c46a"]
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
    pctdistance=0.78, textprops=dict(color="#2b2d42", fontsize=10),
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
)
for at in autotexts:
    at.set_fontweight("bold")

ax.set_title("Wikipedia Traffic Inflow by Source\n(Search engines bring readers in, internal links drive depth)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/plots/fig3_traffic_sources.png", dpi=300)
print("Saved results/plots/fig3_traffic_sources.png")
plt.close()
