"""
preprocess.py - Preprocessing and Feature Extraction for Business Entity Resolution

Observations from EDA (exploration_output.txt):
1. Legal Suffix & Corporate Variations:
   Trailing legal terms (Pvt, Private, Ltd, Limited, Inc, LLC, Corp) vary extensively across
   sources for the same entity, often appearing abbreviated, fully spelled out, in brackets,
   or omitted entirely. Corporate domains (e.g., 'teamaire.com', 'cardiologymetrocare.com')
   are also occasionally used as names. Stripping trailing legal suffixes isolates the true 'core name'.
2. Address Noise, Permutations & Missing PINs:
   Address tokens are frequently transposed (e.g. city/state appearing before street names),
   and components like postal/PIN codes or apartment/unit numbers are often present in Source 2/3
   but omitted in Source 1. Standardizing road/street abbreviations and extracting house numbers
   and postal codes provides invariant structural signals.
3. Multilingual Scripts & Transliteration:
   Training data contains Devanagari, Bengali, Telugu, and other Indic scripts alongside English
   for Indian entities, with phonetic transliterations and accent variants. Normalizing with unidecode
   and case standardization bridges these cross-script discrepancies.
"""

import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import unidecode


# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Extensible abbreviation dictionary
ABBREVIATIONS: Dict[str, str] = {
    "&": "and",
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "rd": "road",
    "st": "street",
    "dr": "drive",
    "ave": "avenue",
    "blvd": "boulevard",
    "hwy": "highway",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "pl": "place",
    "dept": "department",
    "svc": "service",
    "svcs": "services",
    "mfg": "manufacturing",
    "intl": "international",
    "assoc": "associates",
    "mgmt": "management",
}

# Legal suffixes to strip for core name extraction
LEGAL_WORDS = {
    "private",
    "limited",
    "inc",
    "incorporated",
    "llc",
    "corp",
    "corporation",
    "pvt",
    "ltd",
    "pllc",
    "llp",
    "lp",
    "co",
    "company",
}


def clean_text(text: Any) -> str:
    """
    Cleans text by:
    - handling non-string/null inputs
    - stripping accents using unidecode
    - converting to lowercase
    - replacing punctuation with spaces (retaining & for abbreviation expansion)
    - collapsing multiple consecutive whitespace characters into a single space
    """
    if text is None or pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Transliterate / strip accents
    text = unidecode.unidecode(text).lower()

    # Replace punctuation except alphanumeric, spaces, and & with a space
    text = re.sub(r"[^a-z0-9\s&]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_abbreviations(text: Any) -> str:
    """
    Expands abbreviations in text using the ABBREVIATIONS dictionary.
    Applied after clean_text.
    """
    if not text:
        return ""

    # Replace standalone or attached & with ' and '
    text = re.sub(r"&", " and ", text)
    tokens = text.split()
    expanded = [ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def strip_legal_suffixes(name: Any) -> str:
    """
    Removes trailing legal words (private, limited, inc, llc, corp, corporation, pvt, ltd, etc.)
    to isolate the core business name. Returned separately; does not overwrite the original name.
    """
    if not name or not isinstance(name, str):
        return ""

    tokens = name.split()
    while tokens and tokens[-1].lower() in LEGAL_WORDS:
        tokens.pop()

    # If all tokens were legal words, fallback to original name to avoid empty core name
    return " ".join(tokens) if tokens else name


def extract_address_signals(address: Any) -> Dict[str, Optional[str]]:
    """
    Extracts structured invariant signals from an address string:
    - postal_code: any standalone sequence of 5-6 digits (US ZIP or Indian PIN code)
    - house_number: the first numerical token in the address
    - country: country name if explicitly mentioned in the address text
    Returns a dictionary with None for any signal not found.
    """
    signals: Dict[str, Optional[str]] = {
        "postal_code": None,
        "house_number": None,
        "country": None,
    }

    if not address or pd.isna(address):
        return signals

    addr_str = str(address)

    # 1. Potential postal code (standalone 5 or 6 digits)
    m_postal = re.search(r"\b\d{5,6}\b", addr_str)
    if m_postal:
        signals["postal_code"] = m_postal.group(0)

    # 2. First numerical token (potential house/building number)
    m_house = re.search(r"\b\d+\b", addr_str)
    if m_house:
        signals["house_number"] = m_house.group(0)

    # 3. Country mentioned in address text
    addr_lower = addr_str.lower()
    if re.search(r"\bindia\b", addr_lower):
        signals["country"] = "India"
    elif re.search(r"\b(united states|usa|us)\b", addr_lower):
        signals["country"] = "US"
    elif re.search(r"\bfrance\b", addr_lower):
        signals["country"] = "France"

    return signals


def resolve_path(rel_path_str: str) -> Path:
    """Finds path relative to workspace or business_entity_resolution."""
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


def process_and_save_source(
    input_tsv_path: Path,
    output_parquet_path: Path,
    chunk_size: int = 500_000,
) -> None:
    """
    Reads a source TSV in memory-safe chunks, applies clean_text, expand_abbreviations,
    strip_legal_suffixes, and extract_address_signals, and writes to parquet incrementally.
    """
    print(f"\nProcessing {input_tsv_path.name} -> {output_parquet_path.name}...")
    t0 = time.time()

    # Ensure output directory exists
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing output file if present
    if output_parquet_path.exists():
        output_parquet_path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    total_processed = 0

    reader = pd.read_csv(
        input_tsv_path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
    )

    for chunk_idx, chunk in enumerate(reader, 1):
        t_chunk_start = time.time()

        # 1. Clean and expand business names
        cleaned_names = [
            expand_abbreviations(clean_text(name))
            for name in chunk["business_name"]
        ]

        # 2. Extract core names
        core_names = [strip_legal_suffixes(cn) for cn in cleaned_names]

        # 3. Clean and expand business addresses
        cleaned_addresses = [
            expand_abbreviations(clean_text(addr))
            for addr in chunk["business_address"]
        ]

        # 4. Extract structured address signals
        raw_addresses = chunk["business_address"].tolist()
        postal_codes = []
        house_numbers = []
        detected_countries = []

        for addr in raw_addresses:
            sig = extract_address_signals(addr)
            postal_codes.append(sig["postal_code"])
            house_numbers.append(sig["house_number"])
            detected_countries.append(sig["country"])

        # Construct processed DataFrame chunk
        processed_chunk = pd.DataFrame({
            "entity_id": chunk["entity_id"].values,
            "business_name": chunk["business_name"].values,
            "business_address": chunk["business_address"].values,
            "country": chunk["country"].values,
            "cleaned_name": cleaned_names,
            "core_name": core_names,
            "cleaned_address": cleaned_addresses,
            "postal_code": postal_codes,
            "house_number": house_numbers,
            "detected_country": detected_countries,
        })

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(processed_chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                str(output_parquet_path),
                table.schema,
                compression="snappy",
            )

        writer.write_table(table)
        total_processed += len(chunk)
        t_chunk_end = time.time()
        print(
            f"  [Chunk {chunk_idx}] Processed {len(chunk):,} rows "
            f"(Total: {total_processed:,}) in {t_chunk_end - t_chunk_start:.2f}s "
            f"({len(chunk)/(t_chunk_end - t_chunk_start):.0f} rows/s)"
        )

    if writer is not None:
        writer.close()

    total_time = time.time() - t0
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    print(
        f"Finished {output_parquet_path.name}: {total_processed:,} records in {total_time:.2f}s "
        f"({total_processed/total_time:.0f} rows/s, Size: {file_size_mb:.2f} MB)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="train", choices=["train", "test"])
    args, _ = parser.parse_known_args()
    
    dataset_type = args.dataset_type

    print("=" * 80)
    print("      BUSINESS ENTITY RESOLUTION - PREPROCESSING PIPELINE")
    print("=" * 80)

    # Locate inputs
    s1_path = resolve_path(f"dataset/{dataset_type}/{dataset_type}_source1.tsv")
    s2_path = resolve_path(f"dataset/{dataset_type}/{dataset_type}_source2.tsv")
    s3_path = resolve_path(f"dataset/{dataset_type}/{dataset_type}_source3.tsv")

    # Locate/prepare output directory
    output_dir = resolve_path("dataset_processed")
    if not output_dir.exists():
        # Place next to dataset/ if available, or current directory
        if s1_path.parent.parent.parent.exists():
            output_dir = s1_path.parent.parent.parent / "dataset_processed"
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input Directory : {s1_path.parent}")
    print(f"Output Directory: {output_dir}")

    s1_out = output_dir / f"{dataset_type}_source1_clean.parquet"
    s2_out = output_dir / f"{dataset_type}_source2_clean.parquet"
    s3_out = output_dir / f"{dataset_type}_source3_clean.parquet"

    # Process all three sources
    t_global_start = time.time()
    process_and_save_source(s1_path, s1_out)
    process_and_save_source(s2_path, s2_out)
    process_and_save_source(s3_path, s3_out)

    t_global_end = time.time()
    print("\n" + "=" * 80)
    print(f"All 3 sources successfully processed and saved in {t_global_end - t_global_start:.2f}s!")
    print("Output files:")
    for f in [s1_out, s2_out, s3_out]:
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
