import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load existing PageRank results
df_pr = pd.read_csv("results/pagerank_top.csv")

# Filter top 25 PageRank pages that have a views_position
# and calculate the rank surprise: (views_position - pr_position)
bridges = df_pr.dropna(subset=["views_position"]).head(20).copy()
bridges["rank_jump"] = bridges["views_position"] - bridges["pr_position"]
top_bridges = bridges.sort_values(by="rank_jump", ascending=False).head(8)

# Clean titles
clean_titles = [t.replace("_", " ")[:24] for t in top_bridges["title"]]

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(top_bridges))

bars = ax.barh(y_pos, top_bridges["rank_jump"], color="#e76f51", edgecolor="#a8432a", alpha=0.9)
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_titles, fontsize=10, fontweight="bold")
ax.invert_yaxis()

# Annotate with the exact ranks
for i, (_, row) in enumerate(top_bridges.iterrows()):
    pr_pos = int(row['pr_position'])
    v_pos = int(row['views_position'])
    ax.text(row["rank_jump"] + 15, i, f"PageRank #{pr_pos} vs. Views #{v_pos}", 
            va="center", fontsize=9, color="#2b2d42")

ax.set_xlabel("Rank Discrepancy (Views Rank minus PageRank Rank)", fontweight="bold")
ax.set_title("The Hidden Bridges: Pages with Modest Traffic but Massive Structural Importance\n(High PageRank despite low raw views)", 
             fontsize=11, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("results/plots/fig4_hidden_bridges.png", dpi=300)
print("Saved results/plots/fig4_hidden_bridges.png")
