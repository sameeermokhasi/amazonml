"""
blocking.py - Candidate Pair Generation via Inverted Index Blocking

This script implements token-based blocking for Business Entity Resolution:
1. Loads the cleaned parquet files (train_source1_clean, train_source2_clean, train_source3_clean).
2. Groups records by country to ensure cross-country pairs are never generated.
3. Within each country:
   - Identifies and excludes frequent tokens appearing in > 5% of candidate records.
   - Builds an inverted index mapping tokens (from cleaned name + cleaned address) to candidate entity IDs.
4. For each Source 1 entity:
   - Queries the inverted index for all Source 2 & Source 3 entities sharing >= 1 token.
   - Scores candidates by counting shared tokens.
   - Retains the top 25 candidates per Source 1 entity.
5. Saves results incrementally to candidate_pairs_raw.parquet with columns:
   [source1_entity_id, candidate_entity_id, country, shared_token_count].
6. Processes in memory-safe batches and prints progress every 50,000 entities.
7. Supports a --sample flag (default: 50,000 entities) for rapid prototyping and runtime estimation.
"""

import os
import sys
import time
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resolve_path(rel_path_str: str) -> Path:
    """Finds path relative to workspace root or business_entity_resolution."""
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


def get_tokens(name: str, address: str) -> Set[str]:
    """Extracts unique non-empty whitespace-delimited tokens from name and address."""
    combined = f"{name or ''} {address or ''}"
    return set(combined.split())


def build_country_inverted_index(
    cand_df: pd.DataFrame,
    country: str,
    common_threshold_pct: Optional[float] = None,
    use_postal: bool = True,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], List[str], Set[str], float]:
    """
    Builds an inverted index for candidate records (Source 2 + Source 3) within a country.
    Excludes very common tokens using a country-aware filter tuned to address lengths.
    Builds a secondary index on postal codes for high-precision geographic blocking.
    Returns:
        - inverted_index: mapping token -> list of candidate integer indices
        - postal_index: mapping postal_code -> list of candidate integer indices
        - cand_ids: list mapping candidate integer index -> candidate entity_id string
        - common_tokens: set of excluded high-frequency tokens
        - effective_threshold_pct: the actual threshold percentage used
    """
    n_cands = len(cand_df)
    cand_ids = cand_df["entity_id"].tolist()
    names = cand_df["cleaned_name"].fillna("").tolist()
    addrs = cand_df["cleaned_address"].fillna("").tolist()

    # Country-aware threshold tuned separately per country based on real address/token lengths
    if common_threshold_pct is None:
        # Indian addresses are significantly longer (mean ~15 tokens vs ~9 tokens for US).
        # We sample address lengths to compute empirical average document length
        # and scale the base 5% threshold proportionally to prevent over-filtering in longer addresses.
        sample_lens = [len(get_tokens(n, a)) for n, a in zip(names[:10000], addrs[:10000])]
        avg_doc_len = sum(sample_lens) / max(len(sample_lens), 1)
        # Base 5% calibrated for 10-token document; scales with empirical length
        effective_threshold_pct = max(0.02, min(0.12, 0.05 * (avg_doc_len / 10.0)))
    else:
        sample_lens = [len(get_tokens(n, a)) for n, a in zip(names[:10000], addrs[:10000])]
        avg_doc_len = sum(sample_lens) / max(len(sample_lens), 1)
        effective_threshold_pct = common_threshold_pct

    max_doc_freq = effective_threshold_pct * n_cands

    print(f"\n--- Building Inverted Index for Country: {country} ({n_cands:,} candidate records) ---", flush=True)
    print(f"  Average tokens per record: {avg_doc_len:.1f} | Country-aware threshold: {effective_threshold_pct*100:.2f}% (cutoff > {int(max_doc_freq):,} docs)", flush=True)
    t0 = time.time()

    # Pass 1: Compute document frequency for each token
    doc_freq = Counter()
    for name, addr in zip(names, addrs):
        for tok in get_tokens(name, addr):
            doc_freq[tok] += 1

    common_tokens = {tok for tok, count in doc_freq.items() if count > max_doc_freq}
    t_df = time.time() - t0
    print(
        f"  Total unique tokens: {len(doc_freq):,} | "
        f"Excluded frequent tokens (> {effective_threshold_pct*100:.2f}%): {len(common_tokens)}",
        flush=True,
    )
    if common_tokens:
        top_common = sorted(
            [(t, doc_freq[t], doc_freq[t] / n_cands * 100) for t in common_tokens],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        print("  Sample excluded frequent tokens:", flush=True)
        for t, cnt, pct in top_common:
            print(f"    - '{t}': in {cnt:,} records ({pct:.1f}%)", flush=True)

    # Pass 2: Populate inverted index with candidate integer indices
    t0_idx = time.time()
    inverted_index = defaultdict(list)
    for idx, (name, addr) in enumerate(zip(names, addrs)):
        tokens = get_tokens(name, addr) - common_tokens
        for tok in tokens:
            inverted_index[tok].append(idx)

    # Pass 3: Secondary blocking key on postal code
    postal_index = defaultdict(list)
    if use_postal and "postal_code" in cand_df.columns:
        postals = cand_df["postal_code"].tolist()
        for idx, pcode in enumerate(postals):
            if pcode and pd.notna(pcode):
                p_str = str(pcode).strip()
                if p_str:
                    postal_index[p_str].append(idx)

    t_idx = time.time() - t0_idx
    print(
        f"  Inverted index populated: {len(inverted_index):,} tokens | "
        f"Postal index: {len(postal_index):,} unique postal codes in {t_idx:.2f}s",
        flush=True,
    )

    return inverted_index, postal_index, cand_ids, common_tokens, effective_threshold_pct


def evaluate_blocking_recall(
    output_path: Path,
    target_s1_ids: Set[str],
    gt_path: Optional[Path] = None,
) -> Tuple[float, int, int]:
    """
    Calculates blocking recall: % of true matches that appear anywhere in the candidate list.
    """
    if gt_path is None:
        gt_path = resolve_path("dataset/train/train_ground_truth.tsv")
    if not gt_path.exists():
        print(f"Ground truth not found at {gt_path}; skipping recall evaluation.")
        return 0.0, 0, 0

    print(f"\nComputing blocking recall against ground truth ({gt_path.name})...", flush=True)
    t0_eval = time.time()
    df_cand = pd.read_parquet(output_path, columns=["source1_entity_id", "candidate_entity_id"])
    cand_map = df_cand.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    gt_df = pd.read_csv(gt_path, sep="\t")
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(target_s1_ids)]

    total_true = 0
    captured_true = 0
    for _, row in gt_filtered.iterrows():
        matches = row["matched_entity_ids"]
        if pd.notna(matches) and str(matches).strip():
            true_matches = set(x.strip() for x in str(matches).split(",") if x.strip())
            total_true += len(true_matches)
            cand_set = cand_map.get(row["source1_entity_id"], set())
            captured_true += len(true_matches.intersection(cand_set))

    recall = (captured_true / total_true * 100) if total_true > 0 else 0.0
    print("\n" + "=" * 80, flush=True)
    print(f"                 BLOCKING RECALL EVALUATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Total True Matches in Ground Truth: {total_true:,}", flush=True)
    print(f"True Matches Captured in Candidates: {captured_true:,}", flush=True)
    print(f"==> BLOCKING RECALL: {recall:.4f}% ({captured_true:,} / {total_true:,})", flush=True)
    print(f"Recall evaluation computed in {time.time() - t0_eval:.2f}s", flush=True)
    print("=" * 80 + "\n", flush=True)
    return recall, captured_true, total_true


def process_blocking(
    sample_size: Optional[int] = None,
    target_country: Optional[str] = None,
    output_filename: str = "candidate_pairs_raw.parquet",
    batch_size: int = 50_000,
    top_k: int = 25,
    dataset_type: str = "train",
    use_postal: bool = True,
    common_threshold_pct: Optional[float] = None,
    eval_recall: bool = True,
) -> Path:
    """
    Orchestrates candidate pair generation across countries (or a single target country).
    """
    print("=" * 80, flush=True)
    print("       BUSINESS ENTITY RESOLUTION - CANDIDATE BLOCKING PIPELINE", flush=True)
    print("=" * 80, flush=True)
    if target_country:
        print(f"TARGET COUNTRY: {target_country}", flush=True)
    if sample_size:
        print(f"MODE: SAMPLE RUN on first {sample_size:,} Source 1 entities", flush=True)
    else:
        print("MODE: FULL RUN on Source 1 entities", flush=True)
    print(f"CONFIG: Postal Code Key = {'ENABLED' if use_postal else 'DISABLED'} | Common Token Filter = {'Country-Aware (Dynamic)' if common_threshold_pct is None else f'{common_threshold_pct*100:.2f}% (Manual)'}", flush=True)

    # 1. Locate cleaned parquet files
    s1_path = resolve_path(f"dataset_processed/{dataset_type}_source1_clean.parquet")
    s2_path = resolve_path(f"dataset_processed/{dataset_type}_source2_clean.parquet")
    s3_path = resolve_path(f"dataset_processed/{dataset_type}_source3_clean.parquet")

    for label, p in [("Source 1", s1_path), ("Source 2", s2_path), ("Source 3", s3_path)]:
        if not p.exists():
            sys.exit(f"Error: Cleaned parquet not found at {p}. Run preprocess.py first.")
        print(f"  Found {label}: {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)", flush=True)

    # 2. Output path resolution
    output_dir = s1_path.parent
    output_path = output_dir / output_filename
    root_output_path = Path(output_filename)

    # Clean existing file
    if output_path.exists():
        output_path.unlink()

    # Define PyArrow schema for candidate pairs
    schema = pa.schema([
        ("source1_entity_id", pa.string()),
        ("candidate_entity_id", pa.string()),
        ("country", pa.string()),
        ("shared_token_count", pa.int32()),
    ])

    writer = pq.ParquetWriter(str(output_path), schema, compression="snappy")

    # 3. Load Source 1 entities
    cols = ["entity_id", "country", "cleaned_name", "cleaned_address"]
    # Check if parquet has postal_code
    s1_schema = pq.read_schema(str(s1_path))
    if "postal_code" in s1_schema.names:
        cols.append("postal_code")

    print(f"\nLoading Source 1 entities...", flush=True)
    t0_load = time.time()
    df_s1 = pd.read_parquet(s1_path, columns=cols)
    if target_country:
        df_s1 = df_s1[df_s1["country"] == target_country].reset_index(drop=True)
    if sample_size and sample_size < len(df_s1):
        df_s1 = df_s1.iloc[:sample_size].copy()
    print(f"Loaded {len(df_s1):,} Source 1 entities in {time.time() - t0_load:.2f}s", flush=True)
    print(f"Source 1 country breakdown:\n{df_s1['country'].value_counts().to_string()}", flush=True)

    # 4. Load Source 2 & Source 3 candidate records
    print("\nLoading Source 2 and Source 3 candidate records...", flush=True)
    t0_cands = time.time()
    df_s2 = pd.read_parquet(s2_path, columns=cols)
    if target_country:
        df_s2 = df_s2[df_s2["country"] == target_country]
    df_s3 = pd.read_parquet(s3_path, columns=cols)
    if target_country:
        df_s3 = df_s3[df_s3["country"] == target_country]
    df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    print(f"Loaded {len(df_candidates):,} total candidate records in {time.time() - t0_cands:.2f}s", flush=True)

    # 5. Process country by country
    countries = [target_country] if target_country else df_s1["country"].unique().tolist()

    total_s1_processed = 0
    total_pairs_generated = 0
    t_blocking_start = time.time()

    for country in countries:
        # Filter queries and candidates for this country
        s1_country = df_s1[df_s1["country"] == country].reset_index(drop=True)
        cands_country = df_candidates[df_candidates["country"] == country].reset_index(drop=True)

        n_s1_country = len(s1_country)
        print(f"\n=======================================================")
        print(f"Processing Country: {country} ({n_s1_country:,} S1 queries vs {len(cands_country):,} candidates)")
        print(f"=======================================================")

        # Build country inverted index and postal index
        inv_idx, postal_idx, cand_ids, common_tokens, effective_thresh = build_country_inverted_index(
            cands_country, country, common_threshold_pct=common_threshold_pct, use_postal=use_postal
        )

        s1_ids = s1_country["entity_id"].tolist()
        s1_names = s1_country["cleaned_name"].fillna("").tolist()
        s1_addrs = s1_country["cleaned_address"].fillna("").tolist()
        s1_postals = s1_country["postal_code"].tolist() if "postal_code" in s1_country.columns else [None] * n_s1_country

        # Query in batches
        print(f"\nQuerying {n_s1_country:,} Source 1 entities in batches of {batch_size:,}...")
        t_country_query_start = time.time()
        country_pairs = 0

        for b_start in range(0, n_s1_country, batch_size):
            b_end = min(b_start + batch_size, n_s1_country)
            b_s1_ids = s1_ids[b_start:b_end]
            b_s1_names = s1_names[b_start:b_end]
            b_s1_addrs = s1_addrs[b_start:b_end]
            b_s1_postals = s1_postals[b_start:b_end]

            batch_s1_col = []
            batch_cand_col = []
            batch_country_col = []
            batch_score_col = []

            for s1_eid, name, addr, pcode in zip(b_s1_ids, b_s1_names, b_s1_addrs, b_s1_postals):
                q_tokens = get_tokens(name, addr) - common_tokens
                scores = Counter()

                # 1. Primary blocking key: shared tokens
                if q_tokens:
                    for t in q_tokens:
                        if t in inv_idx:
                            scores.update(inv_idx[t][:1000])

                # 2. Secondary blocking key: postal code
                # Entities sharing a postal code are always retrieved as candidates even if name tokens don't overlap
                if use_postal and pcode and pd.notna(pcode):
                    p_str = str(pcode).strip()
                    if p_str and p_str in postal_idx:
                        for cid in postal_idx[p_str]:
                            scores[cid] += 100

                if not scores:
                    continue

                top_matches = scores.most_common(top_k)
                for cid, score in top_matches:
                    batch_s1_col.append(s1_eid)
                    batch_cand_col.append(cand_ids[cid])
                    batch_country_col.append(country)
                    batch_score_col.append(score)

            # Write batch to parquet
            if batch_s1_col:
                batch_table = pa.Table.from_arrays(
                    [
                        pa.array(batch_s1_col, type=pa.string()),
                        pa.array(batch_cand_col, type=pa.string()),
                        pa.array(batch_country_col, type=pa.string()),
                        pa.array(batch_score_col, type=pa.int32()),
                    ],
                    schema=schema,
                )
                writer.write_table(batch_table)
                country_pairs += len(batch_s1_col)
                total_pairs_generated += len(batch_s1_col)

            total_s1_processed += (b_end - b_start)

            # Progress milestone
            elapsed_country = time.time() - t_country_query_start
            q_rate = (b_end) / max(elapsed_country, 0.001)
            print(
                f"  [{country}] Processed {b_end:,} / {n_s1_country:,} S1 entities "
                f"({b_end/n_s1_country*100:.1f}%) | "
                f"Pairs generated: {country_pairs:,} | "
                f"Speed: {q_rate:.1f} queries/s"
            )

            # Check for 50,000 overall entity progress printing
            if total_s1_processed % 50_000 == 0:
                elapsed_total = time.time() - t_blocking_start
                overall_rate = total_s1_processed / max(elapsed_total, 0.001)
                print(
                    f"\n>>> [PROGRESS MILESTONE] Total {total_s1_processed:,} Source 1 entities processed "
                    f"({total_pairs_generated:,} candidate pairs) | "
                    f"Overall Time: {elapsed_total:.1f}s | "
                    f"Overall Rate: {overall_rate:.1f} entities/s <<<\n"
                )

    writer.close()
    t_total = time.time() - t_blocking_start

    # Copy / link to root if executed from root
    if output_path.resolve() != root_output_path.resolve():
        try:
            import shutil
            shutil.copy2(str(output_path), str(root_output_path))
            print(f"Also copied output to: {root_output_path.resolve()}")
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("                     BLOCKING RUN COMPLETED")
    print("=" * 80)
    print(f"Output File              : {output_path}")
    print(f"File Size                : {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Source 1 Entities Queried: {total_s1_processed:,}")
    print(f"Candidate Pairs Produced : {total_pairs_generated:,} (avg {total_pairs_generated/max(total_s1_processed,1):.1f} per entity)")
    print(f"Total Time Taken         : {t_total:.2f}s ({t_total/60:.2f} minutes)")
    overall_speed = total_s1_processed / max(t_total, 0.001)
    print(f"Overall Processing Speed : {overall_speed:.1f} Source 1 entities / sec")

    # Project full runtime
    total_training_s1 = 2_206_821
    projected_seconds = total_training_s1 / max(overall_speed, 0.001)
    projected_hours = projected_seconds / 3600
    print("\n" + "-" * 80)
    print(f"FULL DATASET RUNTIME PROJECTION ({total_training_s1:,} Source 1 entities):")
    print(f"  Projected Time : {projected_seconds:.0f} seconds = {projected_hours:.2f} hours")
    if projected_hours > 1.0:
        print(f"  [FLAGGED]: Projected runtime is {projected_hours:.2f} hours (> 1.0 hour limit)!")
        print("  Full-scale blocking run will take significant time; recommend indexing optimizations or multi-worker clustering.")
    else:
        print(f"  [OK]: Projected runtime is {projected_hours:.2f} hours (<= 1.0 hour limit).")
    print("-" * 80)

    # Evaluate recall against ground truth if train dataset
    gt_path = resolve_path("dataset/train/train_ground_truth.tsv")
    if dataset_type == "train" and eval_recall and gt_path.exists():
        evaluate_blocking_recall(output_path, set(df_s1["entity_id"]), gt_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Candidate Blocking via Inverted Index")
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Target country to process (e.g. India or US). Defaults to all countries.",
    )
    parser.add_argument(
        "--sample",
        nargs="?",
        const=50000,
        type=int,
        default=None,
        help="Process only a sample of Source 1 entities (defaults to 50,000 if flag is specified without an integer)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="candidate_pairs_raw.parquet",
        help="Output parquet filename (default: candidate_pairs_raw.parquet)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="Batch size of Source 1 entities per iteration (default: 50,000)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="Maximum candidate records to keep per Source 1 entity (default: 25)",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset type to process (train or test)",
    )
    parser.add_argument(
        "--no_postal",
        action="store_true",
        help="Disable postal code secondary blocking key",
    )
    parser.add_argument(
        "--common_threshold",
        type=float,
        default=None,
        help="Override country-aware common token threshold (e.g. 0.05 for 5%)",
    )
    parser.add_argument(
        "--yesterday",
        action="store_true",
        help="Run yesterday's baseline configuration (no postal key, 5% fixed threshold)",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip automatic recall evaluation against ground truth",
    )

    args = parser.parse_args()

    use_postal = not args.no_postal
    common_threshold = args.common_threshold
    if args.yesterday:
        use_postal = False
        common_threshold = 0.05

    process_blocking(
        sample_size=args.sample,
        target_country=args.country,
        output_filename=args.output,
        batch_size=args.batch_size,
        top_k=args.top_k,
        dataset_type=args.dataset_type,
        use_postal=use_postal,
        common_threshold_pct=common_threshold,
        eval_recall=not args.no_eval,
    )


if __name__ == "__main__":
    main()

