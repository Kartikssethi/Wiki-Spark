"""Draw every chart for the project from the CSVs in results/ (no Spark needed).

Charts are written to one folder per analysis:
    charts/01_overview        charts/05_pagerank
    charts/02_traffic_sources charts/06_two_way_links
    charts/03_rabbit_hole     charts/07_trends
    charts/04_case_study

Run:  python make_all_charts.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

RESULTS = "results"
OUT = "charts"
SEED_PAGE = "Tom_Holland"  # must match SEED_PAGE in two_way_links.py

# Colors follow meaning across all charts: search = blue, internal links = orange,
# direct = aqua, PageRank = aqua, everything else = yellow
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

MONTH_NAMES = {f"2026-{m:02d}": name for m, name in
               enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"], start=1)}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": INK_2,
    "ytick.color": INK,
    "figure.facecolor": "white",
})


def few_ticks(ax, n=5):
    ax.xaxis.set_major_locator(MaxNLocator(n))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}.csv"))


def folder(name):
    path = os.path.join(OUT, name)
    os.makedirs(path, exist_ok=True)
    return path


def nice(title, width=40):
    """Wikipedia_Title -> 'Wikipedia Title', shortened if very long"""
    title = str(title).replace("_", " ")
    return title if len(title) <= width else title[:width - 1] + "…"


def human(x):
    """1234567 -> '1.2M'"""
    x = float(x)
    for size, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(x) >= size:
            return f"{x / size:.1f}{suffix}"
    return f"{x:.0f}"


def titles(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def hbar(labels, values, title, xlabel, path, color=BLUE, texts=None, subtitle=None, axis_fmt=human):
    """Horizontal bar chart, largest at the top, with a value label at the end of each bar"""
    values = list(values)
    texts = texts if texts is not None else [axis_fmt(v) for v in values]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(values) + 1.7))
    y = range(len(values))
    ax.barh(y, values, color=color, height=0.68)
    ax.set_yticks(list(y))
    ax.set_yticklabels([nice(l) for l in labels])
    ax.invert_yaxis()
    xmax = max(values)
    for i, (v, t) in enumerate(zip(values, texts)):
        ax.text(v + xmax * 0.01, i, t, va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, xmax * 1.28)
    few_ticks(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: axis_fmt(v)))
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    titles(ax, title, subtitle)
    save(fig, path)


def butterfly(labels, left, right, left_label, right_label, title, path, subtitle=None):
    """Two-direction bar chart: left bars = one direction, right bars = the other"""
    left, right = list(left), list(right)
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(labels) + 1.9))
    y = range(len(labels))
    ax.barh(y, [-v for v in left], color=BLUE, height=0.68, label=left_label)
    ax.barh(y, right, color=ORANGE, height=0.68, label=right_label)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, linewidth=0.8)
    lim = max(max(left), max(right)) * 1.12
    ax.set_xlim(-lim, lim)
    few_ticks(ax, 7)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: human(abs(v))))
    ax.set_xlabel("Link clicks, Jan–Sep 2026")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    titles(ax, title, subtitle)
    save(fig, path)


pct = lambda v: f"{v:.1f}%"

# ---------------------------------------------------------------------------
# 01 Overview
# ---------------------------------------------------------------------------
out = folder("01_overview")
sources_monthly = load("story_sources_monthly")

monthly = sources_monthly.groupby("month", as_index=False)["clicks"].sum().sort_values("month")
fig, ax = plt.subplots(figsize=(9, 4.5))
x = [MONTH_NAMES[m] for m in monthly["month"]]
ax.bar(x, monthly["clicks"], color=BLUE, width=0.62)
for i, v in enumerate(monthly["clicks"]):
    ax.text(i, v * 1.01, human(v), ha="center", va="bottom", fontsize=8.5, color=INK_2)
ax.set_ylim(0, monthly["clicks"].max() * 1.12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
ax.set_ylabel("Clicks")
ax.grid(axis="x", visible=False)
titles(ax, "Total clickstream traffic per month", "English Wikipedia, 2026 · all sources")
save(fig, f"{out}/monthly_total_clicks.png")

views_top = load("views_top")
top_views = views_top.head(15)
hbar(top_views["title"], top_views["total_views"],
     "Most-viewed pages, Jan–Sep 2026", "Total views", f"{out}/top_pages_by_views.png")

# ---------------------------------------------------------------------------
# 02 Traffic sources
# ---------------------------------------------------------------------------
out = folder("02_traffic_sources")
sources = load("story_sources").sort_values("share_pct", ascending=False)
hbar(sources["source"], sources["share_pct"],
     "Where Wikipedia readers come from", "Share of all clicks (%)", f"{out}/source_share.png",
     texts=[f"{p:.1f}%  ({human(c)})" for p, c in zip(sources["share_pct"], sources["clicks"])],
     axis_fmt=lambda v: f"{v:.0f}%", subtitle="Jan–Sep 2026 · ~54 billion clicks")

main = {"Search engine": BLUE, "Internal link": ORANGE, "Direct / no referrer": AQUA}
sm = sources_monthly.copy()
sm["group"] = sm["source"].where(sm["source"].isin(main), "Everything else")
share = sm.groupby(["month", "group"])["share_pct"].sum().unstack().sort_index()
fig, ax = plt.subplots(figsize=(9.5, 5))
x = [MONTH_NAMES[m] for m in share.index]
for name, color in list(main.items()) + [("Everything else", YELLOW)]:
    ax.plot(x, share[name], color=color, linewidth=2, marker="o", markersize=6, label=name)
    ax.text(len(x) - 1 + 0.15, share[name].iloc[-1], f"{name} {share[name].iloc[-1]:.0f}%",
            va="center", fontsize=8.5, color=INK)
ax.set_ylim(0, 50)
ax.set_xlim(-0.3, len(x) + 1.6)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("Share of the month's clicks")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4, frameon=False, fontsize=9)
titles(ax, "The traffic mix is stable month to month",
       "Search stays at 42–45% and internal links at 34–36% in every month")
save(fig, f"{out}/source_share_by_month.png")

# ---------------------------------------------------------------------------
# 03 Search vs rabbit hole
# ---------------------------------------------------------------------------
out = folder("03_rabbit_hole")
doors = load("story_search_doors").head(15)
holes = load("story_rabbit_hole_pages").head(15)
gates = load("story_gateways").head(15)

hbar(doors["page"], doors["search_pct"],
     "Search doors: pages people reach straight from Google", "% of views from search engines",
     f"{out}/search_doors.png", color=BLUE, axis_fmt=lambda v: f"{v:.0f}%",
     texts=[pct(v) for v in doors["search_pct"]], subtitle="Pages with at least 1M views, Jan–Sep 2026")

hbar(holes["page"], holes["links_pct"],
     "Rabbit-hole pages: reached by clicking around Wikipedia", "% of views from internal links",
     f"{out}/rabbit_hole_pages.png", color=ORANGE, axis_fmt=lambda v: f"{v:.0f}%",
     texts=[pct(v) for v in holes["links_pct"]], subtitle="Pages with at least 1M views, Jan–Sep 2026")

hbar(gates["page"], gates["onward_per_view"],
     "Gateways: pages that send readers onward", "Onward link clicks per view",
     f"{out}/gateways.png", color=ORANGE, axis_fmt=lambda v: f"{v:.1f}",
     texts=[f"{v:.2f}" for v in gates["onward_per_view"]],
     subtitle="Above 1.0 = each visit leads to more than one further article on average")

fig, ax = plt.subplots(figsize=(8, 6.5))
for data, color, label in [(doors, BLUE, "Search doors"), (holes, ORANGE, "Rabbit-hole pages"),
                           (gates, AQUA, "Gateways")]:
    ax.scatter(data["search_pct"], data["links_pct"], s=60, color=color, label=label,
               edgecolors="white", linewidths=1.5)
ax.set_xlim(-3, 103)
ax.set_ylim(-3, 103)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("% of views from search engines")
ax.set_ylabel("% of views from internal links")
ax.legend(frameon=False, loc="upper right")
titles(ax, "Three kinds of pages", "Each dot is one page from the three top-15 lists")
save(fig, f"{out}/page_types_scatter.png")

# ---------------------------------------------------------------------------
# 04 Case study
# ---------------------------------------------------------------------------
out = folder("04_case_study")
summary = load("story_case_summary").iloc[0]
case = summary["page"]
case_note = (f"{human(summary['views'])} views · {summary['search_pct']:.0f}% from search · "
             f"{summary['links_pct']:.0f}% from links · {summary['onward_per_view']:.2f} onward clicks per view")

inbound = load("story_case_inbound")
outside_names = {"other-search": "Search engines", "other-empty": "Direct / no referrer",
                 "other-internal": "Other Wikimedia sites", "other-external": "Other websites",
                 "other-other": "Other"}
outside = inbound[inbound["prev"].isin(outside_names)]
articles = inbound[~inbound["prev"].isin(outside_names)].head(10)
article_labels = [nice(p, 30) + (" (no direct link)" if t == "other" else "")
                  for p, t in zip(articles["prev"], articles["type"])]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
for ax, labels, values, color, sub in [
        (a1, [outside_names[p] for p in outside["prev"]], outside["clicks"], BLUE, "From outside Wikipedia"),
        (a2, article_labels, articles["clicks"], ORANGE, "From other Wikipedia articles")]:
    values = list(values)
    ax.barh(range(len(values)), values, color=color, height=0.68)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, human(v), va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, max(values) * 1.2)
    few_ticks(ax, 4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
    ax.grid(axis="y", visible=False)
    ax.set_title(sub, loc="left", fontsize=10.5, color=INK_2)
fig.suptitle(f"Where readers of {nice(case)} come from", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.01, 0.905, case_note + " · note the different scales", fontsize=9, color=INK_2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(f"{out}/case_inbound.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", f"{out}/case_inbound.png")

outbound = load("story_case_outbound")
hbar(outbound["curr"], outbound["clicks"],
     f"Where readers of {nice(case)} go next", "Link clicks", f"{out}/case_outbound.png",
     color=ORANGE, subtitle=case_note)

# ---------------------------------------------------------------------------
# 05 PageRank
# ---------------------------------------------------------------------------
out = folder("05_pagerank")
pr_top = load("pagerank_top")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))
for ax, data, col, color, sub, fmt in [
        (a1, top_views, "total_views", BLUE, "Top 15 by raw views (popular)", human),
        (a2, pr_top.head(15), "pagerank", AQUA, "Top 15 by PageRank (central)", lambda v: f"{v * 1e5:.1f}")]:
    values = list(data[col])
    ax.barh(range(len(values)), values, color=color, height=0.68)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([nice(t, 30) for t in data["title"]])
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, fmt(v), va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, max(values) * 1.18)
    few_ticks(ax, 5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, f=fmt: f(v)))
    ax.grid(axis="y", visible=False)
    ax.set_title(sub, loc="left", fontsize=10.5, color=INK_2)
a1.set_xlabel("Total views")
a2.set_xlabel("PageRank score (× 10⁻⁵)")
fig.suptitle("Popular is not the same as central", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.01, 0.905, "Views count every arrival (mostly search). PageRank only follows link clicks inside Wikipedia.", fontsize=9, color=INK_2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(f"{out}/views_vs_pagerank_top15.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", f"{out}/views_vs_pagerank_top15.png")

# Hidden bridges: high PageRank but far down the views ranking
bridges = pr_top.head(100).copy()
bridges["views_pos"] = bridges["views_position"].fillna(5001)
bridges["gap"] = bridges["views_pos"] - bridges["pr_position"]
bridges = bridges.sort_values("gap", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 0.4 * len(bridges) + 1.9))
y = range(len(bridges))
ax.hlines(list(y), bridges["pr_position"], bridges["views_pos"], color=GRID, linewidth=3)
ax.scatter(bridges["pr_position"], list(y), color=AQUA, s=70, zorder=3, label="PageRank position",
           edgecolors="white", linewidths=1.5)
ax.scatter(bridges["views_pos"], list(y), color=BLUE, s=70, zorder=3, label="Views position",
           edgecolors="white", linewidths=1.5)
for i, (p, v) in enumerate(zip(bridges["pr_position"], bridges["views_position"])):
    label = f"#{int(v):,}" if pd.notna(v) else "not in top 5,000"
    ax.text(bridges["views_pos"].iloc[i] * 1.12, i, label, va="center", fontsize=8.5, color=INK_2)
ax.set_xscale("log")
ax.set_xlim(0.7, 30000)
ax.set_yticks(list(y))
ax.set_yticklabels([nice(t) for t in bridges["title"]])
ax.invert_yaxis()
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"#{v:,.0f}"))
ax.set_xlabel("Ranking position (log scale, #1 = top)")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), ncol=2, frameon=False, fontsize=9)
titles(ax, "Hidden bridges: central pages that few people search for",
       "Pages in the PageRank top 100 with the biggest gap to their views ranking")
save(fig, f"{out}/hidden_bridges.png")

scatter = pr_top[pr_top["total_views"] > 0]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.scatter(scatter["total_views"], scatter["pagerank"], s=9, alpha=0.35, color=AQUA, edgecolors="none")
for _, r in scatter.head(8).iterrows():
    ax.annotate(nice(r["title"], 28), (r["total_views"], r["pagerank"]), fontsize=8, color=INK,
                xytext=(5, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
ax.set_xlabel("Total views (log scale)")
ax.set_ylabel("PageRank score (log scale)")
titles(ax, "Popularity vs centrality", "Top 5,000 pages by PageRank · labels = top 8 by PageRank")
save(fig, f"{out}/views_vs_pagerank_scatter.png")

# ---------------------------------------------------------------------------
# 06 Two-way links
# ---------------------------------------------------------------------------
out = folder("06_two_way_links")


def pair_chart(name, title, subtitle, n=15):
    data = load(name).head(n)
    labels = [f"{nice(a, 26)}  ↔  {nice(b, 26)}" for a, b in zip(data["page_a"], data["page_b"])]
    butterfly(labels, data["a_to_b"], data["b_to_a"], "Left page → right page", "Right page → left page",
              title, f"{out}/{name.replace('twoway_', '')}.png", subtitle)


pair_chart("twoway_top_pairs", "Strongest two-way links",
           "Page pairs that readers click between in both directions (each direction ≥ 10K clicks)")
pair_chart("twoway_balanced_pairs", "Truly mutual pairs",
           "Traffic is nearly equal both ways (balance ≥ 0.8): the two pages belong to one topic")
pair_chart("twoway_lopsided_pairs", "Lopsided pairs",
           "Linked both ways, but one direction is at least 5× stronger")

hubs = load("twoway_cluster_hubs").head(20)
hbar(hubs["page"], hubs["two_way_partners"], "Cluster hubs: pages with the most two-way partners",
     "Number of two-way partners", f"{out}/cluster_hubs.png", color=ORANGE,
     axis_fmt=lambda v: f"{v:.0f}", texts=[str(v) for v in hubs["two_way_partners"]],
     subtitle="Each partner is a page readers move to and from in both directions")

seed = load("twoway_seed_cluster")
butterfly([nice(p, 34) for p in seed["partner"]], seed["from_seed"], seed["to_seed"],
          f"{nice(SEED_PAGE)} → partner", f"Partner → {nice(SEED_PAGE)}",
          f"The {nice(SEED_PAGE)} cluster", f"{out}/seed_cluster.png",
          "Every page linked to it strongly in both directions")

# ---------------------------------------------------------------------------
# 07 Trends and spikes
# ---------------------------------------------------------------------------
out = folder("07_trends")


def month_label(page, month):
    return f"{nice(page, 34)}  ({MONTH_NAMES[month]})"


timeline = load("trend_monthly_timeline")
timeline = timeline[timeline["rank"] == 1].sort_values("month")
hbar([f"{MONTH_NAMES[m]}  ·  {nice(p, 32)}" for m, p in zip(timeline["month"], timeline["page"])],
     timeline["jump"], "2026 in Wikipedia: the biggest riser of each month",
     "Increase in views vs the month before", f"{out}/monthly_timeline.png", color=BLUE,
     texts=[f"+{human(v)}" for v in timeline["jump"]],
     subtitle="Found automatically from the data")

spikes = load("trend_biggest_spikes")
spikes = spikes.head(15)
hbar([month_label(p, m) for p, m in zip(spikes["page"], spikes["month"])], spikes["jump"],
     "Biggest one-month spikes", "Increase in views vs the month before", f"{out}/biggest_spikes.png",
     color=BLUE, texts=[f"+{human(j)}  ({human(p)} → {human(v)})"
                        for j, p, v in zip(spikes["jump"], spikes["prev_views"], spikes["views"])])

breakouts = load("trend_breakouts").head(15)
hbar([month_label(p, m) for p, m in zip(breakouts["page"], breakouts["month"])], breakouts["views"],
     "Breakouts: pages that appeared from almost nothing", "Views in the breakout month",
     f"{out}/breakouts.png", color=BLUE,
     texts=[f"{human(v)}  (month before: {human(p)})" for v, p in zip(breakouts["views"], breakouts["prev_views"])],
     subtitle="Highest growth vs the previous month, among months with at least 500K views")

drops = load("trend_biggest_drops")
drops = drops.head(15)
hbar([month_label(p, m) for p, m in zip(drops["page"], drops["month"])], -drops["jump"],
     "Fastest-fading pages", "Drop in views vs the month before", f"{out}/biggest_drops.png",
     color=ORANGE, texts=[f"−{human(-j)}  ({human(p)} → {human(v)})"
                          for j, p, v in zip(drops["jump"], drops["prev_views"], drops["views"])])

wonders = load("trend_one_month_wonders")
wonders = wonders.head(15)
hbar(wonders["page"], wonders["peak_share"] * 100, "One-month wonders",
     "% of all Jan–Sep views that came in the single busiest month", f"{out}/one_month_wonders.png",
     color=BLUE, axis_fmt=lambda v: f"{v:.0f}%",
     texts=[f"{s * 100:.0f}% of {human(t)} views" for s, t in zip(wonders["peak_share"], wonders["total_views"])],
     subtitle="Attention arrived and left within a month")

evergreen = load("trend_evergreen")
evergreen = evergreen.head(15)
hbar(evergreen["page"], evergreen["total_views"], "Evergreen pages: steady all year",
     "Total views, Jan–Sep 2026", f"{out}/evergreen.png", color=AQUA,
     texts=[f"{human(t)}  · quietest month = {s * 100:.0f}% of busiest"
            for t, s in zip(evergreen["total_views"], evergreen["stability"])],
     subtitle="Quietest month is at least 80% of the busiest month")

print("\nAll charts written to", OUT + "/")
