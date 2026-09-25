"""
explore.py - Exploratory Data Analysis for Business Entity Resolution

This script performs 8 comprehensive EDA steps on the training dataset:
1. Loads train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv with sep="\\t".
2. Prints row counts for each file.
3. Prints df.head(10) for each source file.
4. Prints the country value_counts for each source file.
5. Computes and prints the percentage of Source 1 entities in the ground truth with an empty/NaN matched_entity_ids (singleton rate).
6. Computes and prints the distribution of how many matches each non-singleton Source 1 entity has (1, 2, 3+, etc.).
7. Picks 15 random non-empty rows from the ground truth, looks up the Source 1 record and each of its matched Source 2/3 records,
   and prints them side by side (name and address) to visually compare noise patterns.
8. Checks whether any Source 2 or Source 3 entity_id appears in the matched_entity_ids of more than one Source 1 entity,
   and prints how many such cases exist.

Memory-safe design:
- Uses nrows=10 for sample inspection.
- Uses usecols=['country'] for fast country distributions and row counting.
- Uses the full train_ground_truth.tsv (~127 MB, fits comfortably in memory) for exact singleton and distribution metrics.
- Uses targeted single-pass streaming lookups to retrieve records for the 15 sampled entities without loading entire 500MB source files into memory.
"""

import os
import sys
import time
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np


def resolve_file_path(rel_path_str: str) -> Path:
    """
    Resolves relative path across different working directories:
    - Root repo directory (e.g. c:/student_resource)
    - Subdirectory (e.g. c:/student_resource/business_entity_resolution)
    - Script directory (e.g. src/)
    """
    candidates = [
        Path(rel_path_str),
        Path("..") / rel_path_str,
        Path(__file__).resolve().parent.parent / rel_path_str,
        Path(__file__).resolve().parent.parent.parent / rel_path_str,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return Path(rel_path_str).resolve()


def get_tsv_row_count(file_path: Path) -> int:
    """Quickly counts rows in a TSV file excluding header line."""
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            count += 1
    return max(0, count - 1)


def fetch_specific_records(tsv_path: Path, target_ids: set) -> dict:
    """
    Streams a TSV file once to retrieve records for target_ids.
    Memory-safe: does not load the entire file into memory.
    """
    records = {}
    if not target_ids:
        return records

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
        header = header_line.split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        name_idx = header.index("business_name") if "business_name" in header else 1
        addr_idx = header.index("business_address") if "business_address" in header else 2
        country_idx = header.index("country") if "country" in header else 3

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue
            eid = parts[id_idx]
            if eid in target_ids:
                records[eid] = {
                    "entity_id": eid,
                    "business_name": parts[name_idx] if len(parts) > name_idx else "",
                    "business_address": parts[addr_idx] if len(parts) > addr_idx else "",
                    "country": parts[country_idx] if len(parts) > country_idx else "",
                }
                if len(records) == len(target_ids):
                    break
    return records


def main():
    # Ensure stdout/stderr handles UTF-8 on Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 90)
    print("      BUSINESS ENTITY RESOLUTION - DATASET EXPLORATION PIPELINE")
    print("=" * 90)

    # File paths
    s1_path = resolve_file_path("dataset/train/train_source1.tsv")
    s2_path = resolve_file_path("dataset/train/train_source2.tsv")
    s3_path = resolve_file_path("dataset/train/train_source3.tsv")
    gt_path = resolve_file_path("dataset/train/train_ground_truth.tsv")

    print("\n[Step 1] Verifying data file paths:")
    for label, path in [
        ("Source 1", s1_path),
        ("Source 2", s2_path),
        ("Source 3", s3_path),
        ("Ground Truth", gt_path),
    ]:
        exists = path.exists()
        size_mb = (path.stat().st_size / (1024 * 1024)) if exists else 0
        print(f"  - {label:12s}: {str(path)} ({size_mb:.2f} MB) [{'OK' if exists else 'NOT FOUND'}]")
        if not exists:
            sys.exit(f"Error: File not found at {path}")

    # =========================================================================
    # Step 2: Row counts for each file
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Step 2] ROW COUNTS FOR EACH FILE")
    print("=" * 90)
    print("Computing row counts...")

    t0 = time.time()
    s1_count = get_tsv_row_count(s1_path)
    s2_count = get_tsv_row_count(s2_path)
    s3_count = get_tsv_row_count(s3_path)
    gt_count = get_tsv_row_count(gt_path)
    t1 = time.time()

    print(f"\n{'File':<35} | {'Row Count':>15}")
    print("-" * 53)
    print(f"{'train_source1.tsv':<35} | {s1_count:>15,}")
    print(f"{'train_source2.tsv':<35} | {s2_count:>15,}")
    print(f"{'train_source3.tsv':<35} | {s3_count:>15,}")
    print(f"{'train_ground_truth.tsv':<35} | {gt_count:>15,}")
    print("-" * 53)
    print(f"Total Source Records: {s1_count + s2_count + s3_count:>15,}")
    print(f"(Row counts computed in {t1 - t0:.2f}s)")

    # =========================================================================
    # Step 3: Print df.head(10) for each source file
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Step 3] HEAD(10) FOR EACH SOURCE FILE")
    print("=" * 90)

    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 40)

    for label, path in [
        ("Source 1 (Reference)", s1_path),
        ("Source 2", s2_path),
        ("Source 3", s3_path),
    ]:
        print(f"\n>>> {label} - Sample First 10 Rows (nrows=10):")
        df_head = pd.read_csv(path, sep="\t", nrows=10)
        print(df_head.to_string(index=True))

    # =========================================================================
    # Step 4: Country value_counts for each source file
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Step 4] COUNTRY VALUE_COUNTS FOR EACH SOURCE FILE")
    print("=" * 90)

    for label, path in [
        ("Source 1", s1_path),
        ("Source 2", s2_path),
        ("Source 3", s3_path),
    ]:
        t0 = time.time()
        df_country = pd.read_csv(path, sep="\t", usecols=["country"])
        vc = df_country["country"].value_counts(dropna=False)
        pct = (vc / len(df_country) * 100).round(2)
        summary = pd.DataFrame({"Count": vc, "Percentage (%)": pct})
        print(f"\n>>> {label} Country Distribution (Total: {len(df_country):,} records, loaded in {time.time()-t0:.2f}s):")
        print(summary.to_string())

    # =========================================================================
    # Load Ground Truth (Step 5, 6, 8)
    # =========================================================================
    print("\n" + "=" * 90)
    print("Loading full ground truth file for Singleton & Match Distribution analysis...")
    print("=" * 90)
    t0 = time.time()
    df_gt = pd.read_csv(gt_path, sep="\t")
    print(f"Ground truth loaded in {time.time() - t0:.2f}s (Shape: {df_gt.shape[0]:,} rows x {df_gt.shape[1]} cols)")

    # Identify singletons
    # An entity is a singleton if matched_entity_ids is NaN or empty / whitespace string
    is_singleton = df_gt["matched_entity_ids"].isna() | (df_gt["matched_entity_ids"].astype(str).str.strip() == "")
    singleton_count = is_singleton.sum()
    total_gt = len(df_gt)
    non_singleton_count = total_gt - singleton_count
    singleton_rate = (singleton_count / total_gt) * 100

    # =========================================================================
    # Step 5: Singleton rate
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Step 5] SOURCE 1 SINGLETON RATE IN GROUND TRUTH")
    print("=" * 90)
    print(f"Total Source 1 Entities in Ground Truth : {total_gt:>12,}")
    print(f"Singletons (0 matches / empty / NaN)    : {singleton_count:>12,} ({singleton_rate:.4f}%)")
    print(f"Non-Singletons (>= 1 match)             : {non_singleton_count:>12,} ({100 - singleton_rate:.4f}%)")
    print("-" * 55)
    print(f"==> SINGLETON RATE: {singleton_rate:.2f}% (approx {singleton_count:,} out of {total_gt:,})")
    print("-" * 55)

    # =========================================================================
    # Step 6: Distribution of match counts for non-singleton entities
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Step 6] MATCH COUNT DISTRIBUTION FOR NON-SINGLETON SOURCE 1 ENTITIES")
    print("=" * 90)

    df_non_singletons = df_gt[~is_singleton].copy()
    match_counts = df_non_singletons["matched_entity_ids"].apply(lambda x: len(str(x).split(",")))

    dist_counts = match_counts.value_counts().sort_index()
    dist_pct = (dist_counts / len(df_non_singletons) * 100).round(2)
    dist_cum_pct = dist_pct.cumsum().round(2)

    dist_df = pd.DataFrame({
        "Matches": dist_counts.index,
        "Entity Count": dist_counts.values,
        "Percentage (%)": dist_pct.values,
        "Cumulative (%)": dist_cum_pct.values,
    }).set_index("Matches")

    print(f"Distribution across {len(df_non_singletons):,} non-singleton entities:")
    print(dist_df.to_string())

    print("\nSummary Statistics of Match Counts:")
    print(f"  - Minimum matches : {match_counts.min()}")
    print(f"  - Maximum matches : {match_counts.max()}")
    print(f"  - Mean matches    : {match_counts.mean():.2f}")
    print(f"  - Median matches  : {match_counts.median():.0f}")
    print(f"  - Mode matches    : {match_counts.mode().iloc[0]}")

    # =========================================================================
    # Step 7: 15 Random non-empty rows inspected side by side
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Step 7] 15 RANDOM SAMPLES: SIDE-BY-SIDE NOISE PATTERN COMPARISON")
    print("=" * 90)
    print("Sampling 15 random non-empty ground truth rows and fetching Source 1, 2, 3 records...")

    sample_seed = 42
    sample_gt = df_non_singletons.sample(n=15, random_state=sample_seed)

    s1_needed = set(sample_gt["source1_entity_id"])
    s2_needed = set()
    s3_needed = set()

    for m_str in sample_gt["matched_entity_ids"]:
        for eid in str(m_str).split(","):
            eid = eid.strip()
            if eid.startswith("S2-"):
                s2_needed.add(eid)
            elif eid.startswith("S3-"):
                s3_needed.add(eid)

    t_fetch_start = time.time()
    s1_rec_dict = fetch_specific_records(s1_path, s1_needed)
    s2_rec_dict = fetch_specific_records(s2_path, s2_needed)
    s3_rec_dict = fetch_specific_records(s3_path, s3_needed)
    t_fetch_end = time.time()
    print(f"Fetched {len(s1_rec_dict)} S1, {len(s2_rec_dict)} S2, and {len(s3_rec_dict)} S3 records in {t_fetch_end - t_fetch_start:.2f}s.\n")

    for i, (_, row) in enumerate(sample_gt.iterrows(), 1):
        s1_id = row["source1_entity_id"]
        s1_rec = s1_rec_dict.get(s1_id, {"business_name": "NOT_FOUND", "business_address": "NOT_FOUND", "country": "N/A"})
        matched_ids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]

        print("=" * 110)
        print(f"SAMPLE {i:02d} / 15  |  Source 1 Entity ID: {s1_id}  |  Total Matches: {len(matched_ids)}  |  Country: {s1_rec.get('country')}")
        print("-" * 110)
        print(f"{'Source':<10} | {'Entity ID':<14} | {'Business Name':<38} | {'Business Address'}")
        print("-" * 110)

        # Print Reference Source 1
        s1_name = s1_rec.get("business_name", "")
        s1_addr = s1_rec.get("business_address", "")
        print(f"{'S1 (Ref)':<10} | {s1_id:<14} | {s1_name:<38} | {s1_addr}")

        # Print all matches
        for m_idx, mid in enumerate(matched_ids, 1):
            if mid.startswith("S2-"):
                m_rec = s2_rec_dict.get(mid, {})
                label = f"Match {m_idx} (S2)"
            elif mid.startswith("S3-"):
                m_rec = s3_rec_dict.get(mid, {})
                label = f"Match {m_idx} (S3)"
            else:
                m_rec = {}
                label = f"Match {m_idx}"

            m_name = m_rec.get("business_name", "<Not Found>")
            m_addr = m_rec.get("business_address", "<Not Found>")
            print(f"{label:<10} | {mid:<14} | {m_name:<38} | {m_addr}")

        print("-" * 110)

    # =========================================================================
    # Step 8: Multi-mapping check for Source 2 and Source 3
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Step 8] CHECK WHETHER ANY S2 OR S3 ENTITY APPEARS IN MORE THAN ONE S1 ENTITY")
    print("=" * 90)
    print("Analyzing all matched_entity_ids across ground truth...")

    t0 = time.time()
    s2_s3_id_counter = Counter()

    for matched_str in df_non_singletons["matched_entity_ids"]:
        for eid in str(matched_str).split(","):
            eid = eid.strip()
            if eid:
                s2_s3_id_counter[eid] += 1

    t1 = time.time()
    multi_matched = {k: v for k, v in s2_s3_id_counter.items() if v > 1}
    num_multi = len(multi_matched)
    total_unique_s2_s3 = len(s2_s3_id_counter)

    print(f"Total unique Source 2 and Source 3 entity IDs in ground truth: {total_unique_s2_s3:,}")
    print(f"Number of S2/S3 entity IDs matched to > 1 Source 1 entity      : {num_multi}")
    if num_multi > 0:
        print(f"WARNING: Found {num_multi} entity IDs appearing in multiple Source 1 matches!")
        print(f"Sample multi-mapped entity IDs: {list(multi_matched.items())[:10]}")
    else:
        print("==> RESULT: EXACTLY ZERO (0) Source 2 or Source 3 entity IDs map to more than one Source 1 entity!")
        print("==> ARCHITECTURAL INSIGHT: The relationship from S2/S3 to S1 is strictly 1-to-1 (injective).")
        print("    A Source 1 entity can have multiple S2/S3 matches, but each S2/S3 record belongs to AT MOST one S1 entity.")
    print(f"(Multi-mapping check completed in {t1 - t0:.2f}s)")

    print("\n" + "=" * 90)
    print("                      EXPLORATION COMPLETED SUCCESSFULLY")
    print("=" * 90)


if __name__ == "__main__":
    main()
