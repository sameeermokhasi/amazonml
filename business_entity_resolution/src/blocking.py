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
    common_threshold_pct: float = 0.05,
) -> Tuple[Dict[str, List[int]], List[str], Set[str]]:
    """
    Builds an inverted index for candidate records (Source 2 + Source 3) within a country.
    Excludes very common tokens that appear in > 5% of candidate records in that country.
    Returns:
        - inverted_index: mapping token -> list of candidate integer indices
        - cand_ids: list mapping candidate integer index -> candidate entity_id string
        - common_tokens: set of excluded high-frequency tokens
    """
    n_cands = len(cand_df)
    max_doc_freq = common_threshold_pct * n_cands
    cand_ids = cand_df["entity_id"].tolist()
    names = cand_df["cleaned_name"].fillna("").tolist()
    addrs = cand_df["cleaned_address"].fillna("").tolist()

    print(f"\n--- Building Inverted Index for Country: {country} ({n_cands:,} candidate records) ---", flush=True)
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
        f"Excluded frequent tokens (> {common_threshold_pct*100:.0f}% = {int(max_doc_freq):,} docs): {len(common_tokens)}",
        flush=True,
    )
    if common_tokens:
        top_common = sorted(
            [(t, doc_freq[t], doc_freq[t] / n_cands * 100) for t in common_tokens],
            key=lambda x: x[1],
            reverse=True,
        )[:15]
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

    t_idx = time.time() - t0_idx
    print(
        f"  Inverted index populated: {len(inverted_index):,} indexed tokens "
        f"in {t_idx:.2f}s ({n_cands / (t_df + t_idx):.0f} records/s total)",
        flush=True,
    )

    return inverted_index, cand_ids, common_tokens


def process_blocking(
    sample_size: Optional[int] = None,
    target_country: Optional[str] = None,
    output_filename: str = "candidate_pairs_raw.parquet",
    batch_size: int = 50_000,
    top_k: int = 25,
    dataset_type: str = "train",
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

        # Build country inverted index
        inv_idx, cand_ids, common_tokens = build_country_inverted_index(
            cands_country, country, common_threshold_pct=0.005
        )

        s1_ids = s1_country["entity_id"].tolist()
        s1_names = s1_country["cleaned_name"].fillna("").tolist()
        s1_addrs = s1_country["cleaned_address"].fillna("").tolist()

        # Query in batches
        print(f"\nQuerying {n_s1_country:,} Source 1 entities in batches of {batch_size:,}...")
        t_country_query_start = time.time()
        country_pairs = 0

        for b_start in range(0, n_s1_country, batch_size):
            b_end = min(b_start + batch_size, n_s1_country)
            b_s1_ids = s1_ids[b_start:b_end]
            b_s1_names = s1_names[b_start:b_end]
            b_s1_addrs = s1_addrs[b_start:b_end]

            batch_s1_col = []
            batch_cand_col = []
            batch_country_col = []
            batch_score_col = []

            for s1_eid, name, addr in zip(b_s1_ids, b_s1_names, b_s1_addrs):
                q_tokens = get_tokens(name, addr) - common_tokens
                if not q_tokens:
                    continue

                # Find postings for query tokens
                postings_list = [inv_idx[t] for t in q_tokens if t in inv_idx]
                if not postings_list:
                    continue

                if len(postings_list) == 1:
                    # Single token matched: top candidates all have score = 1
                    top_matches = [(cid, 1) for cid in postings_list[0][:top_k]]
                else:
                    # Count occurrences of candidate indices across matched tokens
                    scores = Counter()
                    for p in postings_list:
                        scores.update(p)
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

    args = parser.parse_args()
    process_blocking(
        sample_size=args.sample,
        target_country=args.country,
        output_filename=args.output,
        batch_size=args.batch_size,
        top_k=args.top_k,
        dataset_type=args.dataset_type,
    )


if __name__ == "__main__":
    main()

