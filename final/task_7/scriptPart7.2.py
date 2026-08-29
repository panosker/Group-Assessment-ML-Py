"""
PART 7 - Further Analysis
2)Relationship between Topic and Sentiment


  - cross-tabulation of Topic x Sentiment (counts + row percentages)
  - chi-square test of independence + Cramer's V effect size
  - standardised residuals (which topic-sentiment cells drive the pattern)
  - a net sentiment index per topic
  - three figures saved as PNG for the report
  - annotator-disagreement rates per topic (limitations section)

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures to file instead of opening a window
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
INPUT_FILE = "Final_Concatenated_File_Cleaned.csv"

TOPIC_COL = "Topic"
SENTIMENT_COL = "Final Opinion"
ANNOTATOR_COLS = ["ai_opinion", "Alex", "Panagiotis", "Athina", "Antonia"]

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#4C9F70", "neutral": "#B0B0B0", "negative": "#C1543B"}


# ---------------------------------------------------------------
# 1. Cross-tabulation
# ---------------------------------------------------------------
def build_crosstab(df):
    """Counts and within-topic percentages of each sentiment class."""
    counts = pd.crosstab(df[TOPIC_COL], df[SENTIMENT_COL])
    counts = counts[[c for c in SENTIMENT_ORDER if c in counts.columns]]
    row_pct = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)

    print("=== Topic x Sentiment: counts ===")
    print(counts)
    print("\n=== Topic x Sentiment: row percentages (within each topic) ===")
    print(row_pct)
    return counts, row_pct


# ---------------------------------------------------------------
# 2. Chi-square test of independence + effect size
# ---------------------------------------------------------------
def test_independence(counts):
    """
    H0: sentiment distribution is independent of topic.
    A small p-value means topic and sentiment ARE associated.
    Cramer's V measures HOW STRONG that association is (0 = none, 1 = perfect);
    roughly, 0.1 = weak, 0.3 = moderate, 0.5 = strong.
    """
    chi2, p, dof, expected = chi2_contingency(counts)
    n = counts.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(counts.shape) - 1)))

    print("\n=== Chi-square test of independence ===")
    print(f"chi2 = {chi2:.2f}   dof = {dof}   p = {p:.3e}")
    print(f"Cramer's V = {cramers_v:.3f}")
    print("Conclusion:",
          "topic and sentiment are significantly associated (reject H0)."
          if p < 0.05 else "no significant association (fail to reject H0).")

    expected_df = pd.DataFrame(expected, index=counts.index, columns=counts.columns)
    residuals = (counts - expected_df) / np.sqrt(expected_df)

    print("\n=== Standardised (Pearson) residuals ===")
    print("(> +2 = far MORE posts than expected; < -2 = far FEWER than expected)")
    print(residuals.round(2))

    return {"chi2": chi2, "p": p, "dof": dof, "cramers_v": cramers_v,
            "expected": expected_df, "residuals": residuals}


# ---------------------------------------------------------------
# 3. Net sentiment index per topic
# ---------------------------------------------------------------
def net_sentiment_index(row_pct):
    """
        net = %positive - %negative
    Positive value = topic skews positive overall; negative = skews negative.
    """
    net = (row_pct["positive"] - row_pct["negative"]).sort_values(ascending=False)
    print("\n=== Net sentiment index (%positive - %negative) ===")
    for topic, val in net.items():
        print(f"  {topic:<14s} {val:+6.1f}")
    return net


# ---------------------------------------------------------------
# 4. Visualisations
# ---------------------------------------------------------------
def plot_stacked_percentages(row_pct, filename="fig_topic_sentiment_stacked.png"):
    order = row_pct.mean(axis=1).index.tolist()
    order = row_pct.sort_values("positive", ascending=False).index.tolist()
    data = row_pct.loc[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(data))
    for sentiment in SENTIMENT_ORDER:
        if sentiment not in data.columns:
            continue
        vals = data[sentiment].values
        ax.bar(data.index, vals, bottom=bottom,
               label=sentiment.capitalize(), color=COLORS[sentiment], edgecolor="white")
        for i, v in enumerate(vals):
            if v > 4:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=10)
        bottom += vals

    ax.set_ylabel("Share of posts (%)")
    ax.set_title("Sentiment distribution within each topic")
    ax.set_ylim(0, 100)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_grouped_counts(counts, filename="fig_topic_sentiment_grouped.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(counts.index))
    width = 0.26
    for i, sentiment in enumerate(SENTIMENT_ORDER):
        if sentiment not in counts.columns:
            continue
        ax.bar(x + (i - 1) * width, counts[sentiment], width,
               label=sentiment.capitalize(), color=COLORS[sentiment])

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index)
    ax.set_ylabel("Number of posts")
    ax.set_title("Sentiment counts by topic")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_residual_heatmap(residuals, filename="fig_topic_sentiment_residuals.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vmax = np.abs(residuals.values).max()
    im = ax.imshow(residuals.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(residuals.columns)))
    ax.set_xticklabels([c.capitalize() for c in residuals.columns])
    ax.set_yticks(range(len(residuals.index)))
    ax.set_yticklabels(residuals.index)

    for i in range(residuals.shape[0]):
        for j in range(residuals.shape[1]):
            val = residuals.values[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    color="white" if abs(val) > vmax * 0.55 else "black",
                    fontweight="bold")

    ax.set_title("Standardised residuals\n(red = more posts than expected, blue = fewer)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# ---------------------------------------------------------------
# 5. Annotation reliability per topic (for the limitations discussion)
# ---------------------------------------------------------------
def annotator_disagreement(df):
    """
    No. of annotators and agreement. Per post and per topic.
    """
    present = df[ANNOTATOR_COLS]

    def row_stats(row):
        labels = [str(v).strip().lower() for v in row if pd.notna(v)]
        if not labels:
            return pd.Series({"n_annotators": 0, "unanimous": np.nan})
        return pd.Series({"n_annotators": len(labels),
                          "unanimous": len(set(labels)) == 1})

    stats = present.apply(row_stats, axis=1)
    stats[TOPIC_COL] = df[TOPIC_COL]

    summary = stats.groupby(TOPIC_COL).agg(
        avg_annotators=("n_annotators", "mean"),
        unanimous_rate=("unanimous", "mean"),
    )
    summary["unanimous_rate"] = (summary["unanimous_rate"] * 100).round(1)
    summary["avg_annotators"] = summary["avg_annotators"].round(2)

    print("\n=== Annotation reliability per topic ===")
    print(summary)
    return summary


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=[TOPIC_COL, SENTIMENT_COL])
    df[SENTIMENT_COL] = df[SENTIMENT_COL].str.strip().str.lower()
    print(f"Loaded {len(df)} labelled posts across {df[TOPIC_COL].nunique()} topics.\n")

    counts, row_pct = build_crosstab(df)
    stats = test_independence(counts)
    net = net_sentiment_index(row_pct)

    plot_stacked_percentages(row_pct)
    plot_grouped_counts(counts)
    plot_residual_heatmap(stats["residuals"])

    annotator_disagreement(df)

    # Export tables for the report
    counts.to_csv("topic_sentiment_counts.csv")
    row_pct.to_csv("topic_sentiment_percentages.csv")
    stats["residuals"].round(2).to_csv("topic_sentiment_residuals.csv")
    print("\nSaved topic_sentiment_counts.csv / _percentages.csv / _residuals.csv")