"""
write_output.py - Submission Output Writer and Validator for Business Entity Resolution

This module handles serialization and pre-write validation for challenge submissions:
- matching_results.tsv: Final entity matches scored on leaderboard.
- candidate_pairs.tsv: Blocking candidate shortlist.

Guarantees:
- Tab-separated (.tsv) format.
- Exactly two columns: source1_entity_id and matched_entity_ids (or candidate_entity_ids).
- Comma-joined ID lists with no intra-list duplicates.
- Strict empty strings ("") for zero-match entities (no NaN, null, or "nan").
- No quoting issues or stray quote marks.
- Pre-write validation verifying 100% 1-to-1 Source 1 test coverage with zero duplicates.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Union
import pandas as pd


def check_source1_coverage(
    df: pd.DataFrame,
    test_source1_path: Optional[Union[str, Path]] = None,
    required_ids: Optional[Set[str]] = None,
) -> None:
    """
    Validates that every required Source 1 entity ID is present in df exactly once.
    Raises ValueError with a clear diagnostic message if validation fails.
    """
    if "source1_entity_id" not in df.columns:
        raise ValueError("DataFrame missing required column: 'source1_entity_id'")

    s1_series = df["source1_entity_id"].astype(str).str.strip()

    # 1. Check for duplicate rows
    duplicates = s1_series[s1_series.duplicated()].unique()
    if len(duplicates) > 0:
        sample_dupes = list(duplicates[:5])
        raise ValueError(
            f"Validation Failed: Found {len(duplicates)} duplicate source1_entity_id row(s). "
            f"Each Source 1 entity must have exactly ONE row. Duplicates: {sample_dupes}"
        )

    # 2. Check coverage against required test IDs
    if required_ids is None and test_source1_path is not None:
        p = Path(test_source1_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                next(f, None)  # Skip header
                required_ids = {line.split("\t", 1)[0].strip() for line in f if line.strip()}

    if required_ids is not None:
        output_ids = set(s1_series)
        missing_ids = required_ids - output_ids
        extra_ids = output_ids - required_ids

        if missing_ids:
            sample_missing = sorted(list(missing_ids))[:5]
            raise ValueError(
                f"Validation Failed: {len(missing_ids)} required Source 1 entity ID(s) missing from output. "
                f"Every test entity must have a row (empty for singletons). Missing: {sample_missing}"
            )

        if extra_ids:
            sample_extra = sorted(list(extra_ids))[:5]
            raise ValueError(
                f"Validation Failed: {len(extra_ids)} unknown Source 1 entity ID(s) in output not present "
                f"in test reference. Extra IDs: {sample_extra}"
            )


def validate_id_lists(df: pd.DataFrame, id_col_name: str) -> None:
    """
    Validates the contents of matched/candidate ID lists:
    - No intra-list duplicates
    - No self-matches (S1- prefix)
    - Valid prefixes (must start with S2- or S3-)
    """
    if id_col_name not in df.columns:
        raise ValueError(f"DataFrame missing required ID column: '{id_col_name}'")

    for idx, row in df.iterrows():
        s1_id = str(row["source1_entity_id"])
        val = row[id_col_name]

        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            if len(val) == 0:
                continue
            id_list = [str(x).strip() for x in val if str(x).strip()]
        else:
            if pd.isna(val) or str(val).strip() == "":
                continue
            id_list = [x.strip() for x in str(val).split(",") if x.strip()]

        # Check intra-list duplicates
        if len(id_list) != len(set(id_list)):
            dupes = [x for x in id_list if id_list.count(x) > 1]
            raise ValueError(
                f"Validation Failed at row {idx} ({s1_id}): repeated ID inside {id_col_name}: {set(dupes)}. "
                "No duplicate IDs allowed within a single entity list."
            )

        # Check self-matches and prefixes
        for mid in id_list:
            if mid.startswith("S1-"):
                raise ValueError(
                    f"Validation Failed at row {idx} ({s1_id}): {id_col_name} contains Source 1 self-match '{mid}'. "
                    "Only S2- and S3- entity IDs are allowed."
                )
            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                raise ValueError(
                    f"Validation Failed at row {idx} ({s1_id}): '{mid}' lacks valid S2- or S3- prefix."
                )


def format_id_list(val) -> str:
    """Formats an ID list into a comma-separated string, returning empty string for empty/null."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple, set)):
        if len(val) == 0:
            return ""
        seen = set()
        clean_ids = []
        for x in val:
            item = str(x).strip()
            if item and item not in seen:
                seen.add(item)
                clean_ids.append(item)
        return ",".join(clean_ids)
    if pd.isna(val) or str(val).strip() == "":
        return ""
    return str(val).strip()


def write_results(
    df: pd.DataFrame,
    path: Union[str, Path],
    id_col_name: str = "matched_entity_ids",
    test_source1_path: Optional[Union[str, Path]] = None,
    required_ids: Optional[Set[str]] = None,
    validate_coverage: bool = True,
) -> Path:
    """
    Serializes matches/candidates to a strict tab-separated (.tsv) file.

    Parameters:
        df: DataFrame with 'source1_entity_id' and id_col_name ('matched_entity_ids' or 'candidate_entity_ids')
        path: Target file path
        id_col_name: Column name for IDs ('matched_entity_ids' or 'candidate_entity_ids')
        test_source1_path: Path to test_source1.tsv for coverage check
        required_ids: Explicit set of required Source 1 IDs for coverage check
        validate_coverage: If True, performs pre-write coverage and integrity validation
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if validate_coverage:
        validate_id_lists(df, id_col_name)
        if test_source1_path is not None or required_ids is not None:
            check_source1_coverage(df, test_source1_path, required_ids)

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for _, row in df.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            formatted_ids = format_id_list(row[id_col_name])
            f.write(f"{s1_id}\t{formatted_ids}\n")

    return out_path
