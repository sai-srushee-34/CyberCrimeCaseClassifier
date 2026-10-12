"""
Exploratory Data Analysis (EDA) and Quality Auditing Pipeline.
Analyzes curated_complaints.csv, computes length and lexical statistics,
detects template artifacts and source bias, and generates publication-ready
visualizations in reports/figures/.
"""

import os
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"

CANONICAL_CATEGORIES = [
    "Phishing",
    "Online Financial Fraud",
    "Identity Theft",
    "Account Hacking",
    "Cyberbullying / Online Harassment",
    "Fake Social Media Account",
    "Malware / Ransomware",
    "Online Shopping Fraud",
]


def run_eda(
    data_path: str = "data/processed/curated_complaints.csv",
    fig_dir: str = "reports/figures"
):
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    print("==================================================")
    print("1. DATASET DIMENSIONS & DATA INTEGRITY")
    print("==================================================")
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_vals = df.isnull().sum()
    exact_row_dups = df.duplicated().sum()
    exact_text_dups = df["complaint_text"].duplicated().sum()

    print(f"Total Records: {total_rows:,}")
    print(f"Total Columns: {total_cols}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing Values per Column:\n{missing_vals.to_dict()}")
    print(f"Exact Duplicate Rows: {exact_row_dups}")
    print(f"Exact Duplicate Texts: {exact_text_dups}")

    print("\n==================================================")
    print("2. CLASS DISTRIBUTION")
    print("==================================================")
    cat_counts = df["category"].value_counts()
    for cat in CANONICAL_CATEGORIES:
        cnt = cat_counts.get(cat, 0)
        pct = (cnt / total_rows) * 100
        print(f" - {cat:35s}: {cnt:5,d} ({pct:5.1f}%)")

    print("\n==================================================")
    print("3. SOURCE COMPOSITION & PROVENANCE BIAS")
    print("==================================================")
    src_crosstab = pd.crosstab(df["category"], df["source_name"], margins=True)
    print(src_crosstab)

    synth_count = (df["is_synthetic"] == True).sum()
    real_count = (df["is_synthetic"] == False).sum()
    print(f"\nReal-world Records: {real_count:,} ({real_count/total_rows*100:.1f}%)")
    print(f"Synthetic Records:  {synth_count:,} ({synth_count/total_rows*100:.1f}%)")

    print("\n==================================================")
    print("4. COMPLAINT LENGTH METRICS")
    print("==================================================")
    df["char_len"] = df["complaint_text"].str.len()
    df["word_cnt"] = df["complaint_text"].apply(lambda s: len(str(s).split()))

    stats_by_cat = df.groupby("category")[["word_cnt", "char_len"]].agg(["min", "mean", "median", "max", "std"])
    print(stats_by_cat)

    print("\nOverall Word Count Quantiles:")
    print(df["word_cnt"].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

    print("\n==================================================")
    print("5. TEMPLATE & REPETITIVE PREFIX ANALYSIS")
    print("==================================================")
    for cat in CANONICAL_CATEGORIES:
        grp = df[df["category"] == cat]
        prefixes = grp["complaint_text"].apply(lambda s: " ".join(str(s).split()[:4]))
        top_pref = Counter(prefixes).most_common(3)
        print(f"Category [{cat}] (Total: {len(grp)}):")
        for p, count in top_pref:
            print(f"   {count:4d} ({count/len(grp)*100:4.1f}%): \"{p}\"")

    print("\n==================================================")
    print("6. PRIVACY REDACTION & RESIDUAL SCAN")
    print("==================================================")
    email_pat = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    phone_pat = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    card_pat = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4,7}\b")

    raw_emails = df["complaint_text"].apply(lambda x: len(email_pat.findall(str(x)))).sum()
    raw_phones = df["complaint_text"].apply(lambda x: len(phone_pat.findall(str(x)))).sum()
    raw_cards = df["complaint_text"].apply(lambda x: len(card_pat.findall(str(x)))).sum()

    print(f"Unmasked Email Addresses Detected: {raw_emails}")
    print(f"Unmasked Phone Numbers Detected:   {raw_phones}")
    print(f"Unmasked Card/Account IDs Detected: {raw_cards}")

    print("\nMask Tags Present in Curated Corpus:")
    for tag in ["[USER_REDACTED]", "[URL_REDACTED]", "[EMAIL_REDACTED]", "[PHONE_REDACTED]", "[ACCOUNT_REDACTED]"]:
        c = df["complaint_text"].str.contains(re.escape(tag)).sum()
        print(f"  {tag:20s}: {c:5,d} occurrences")

    # ==================================================
    # 7. GENERATE PUBLICATION-READY CHARTS
    # ==================================================
    print("\nGenerating charts in reports/figures/...")

    # Plot 1: Category Distribution
    plt.figure(figsize=(10, 6))
    colors = ["#2b5c8f" if c != "Cyberbullying / Online Harassment" else "#d95f02" for c in cat_counts.index]
    bars = plt.barh(cat_counts.index, cat_counts.values, color=colors, edgecolor="black", linewidth=0.7)
    plt.xlabel("Number of Complaints", fontweight="bold")
    plt.title("Complaint Category Distribution (Showing Class Imbalance)", fontweight="bold", pad=12)
    plt.gca().invert_yaxis()
    for bar in bars:
        w = bar.get_width()
        pct = (w / total_rows) * 100
        plt.text(w + 50, bar.get_y() + bar.get_height() / 2, f"{int(w):,} ({pct:.1f}%)", va="center", fontsize=9)
    plt.xlim(0, max(cat_counts.values) * 1.15)
    plt.tight_layout()
    cat_fig_path = os.path.join(fig_dir, "category_distribution.png")
    plt.savefig(cat_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {cat_fig_path}")

    # Plot 2: Complaint Length by Category (Boxplot)
    plt.figure(figsize=(12, 6))
    cat_order = list(cat_counts.index)
    sns.boxplot(
        data=df,
        y="category",
        x="word_cnt",
        order=cat_order,
        hue="category",
        legend=False,
        palette="Blues_r",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 5},
        fliersize=2
    )
    plt.xlabel("Word Count per Complaint", fontweight="bold")
    plt.ylabel("")
    plt.title("Complaint Length (Word Count) Distribution by Category", fontweight="bold", pad=12)
    plt.xlim(0, 160)
    plt.tight_layout()
    len_fig_path = os.path.join(fig_dir, "length_distribution_by_category.png")
    plt.savefig(len_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {len_fig_path}")

    # Plot 3: Source Composition by Category (Stacked Bar Chart)
    plt.figure(figsize=(11, 6))
    source_pivot = pd.crosstab(df["category"], df["source_name"])
    ax = source_pivot.plot(
        kind="barh",
        stacked=True,
        figsize=(11, 6),
        color=["#2b5c8f", "#7570b3"],
        edgecolor="black",
        linewidth=0.6
    )
    plt.xlabel("Complaint Count", fontweight="bold")
    plt.ylabel("")
    plt.title("Source Provenance Composition by Category", fontweight="bold", pad=12)
    plt.legend(["Synthetic Scaffold", "Real Twitter Harassment"], loc="lower right", frameon=True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    src_fig_path = os.path.join(fig_dir, "source_distribution_by_category.png")
    plt.savefig(src_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {src_fig_path}")

    # Plot 4: Overall Word Count Histogram / KDE
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="word_cnt",
        hue="is_synthetic",
        palette={True: "#2b5c8f", False: "#d95f02"},
        bins=40,
        kde=True,
        element="step",
        stat="density",
        common_norm=False
    )
    plt.xlabel("Word Count", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.title("Word Count Density: Synthetic Complaints vs. Real Twitter Harassment", fontweight="bold", pad=12)
    plt.legend(["Real Social Media", "Synthetic Scaffold"], loc="upper right")
    plt.xlim(0, 150)
    plt.tight_layout()
    hist_fig_path = os.path.join(fig_dir, "complaint_length_histogram.png")
    plt.savefig(hist_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {hist_fig_path}")

    print("\nEDA pipeline execution complete.")


if __name__ == "__main__":
    run_eda()
