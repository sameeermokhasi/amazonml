"""
features.py - Feature Engineering Pipeline for Business Entity Resolution

This module computes pairwise matching features between Source 1 reference entities
and candidate entities (from Source 2 or Source 3) identified during the blocking phase.

Features generated for each candidate pair:
1. name_token_sort_ratio: RapidFuzz token sort ratio between Source 1 name and candidate name.
2. name_token_set_ratio: RapidFuzz token set ratio between Source 1 name and candidate name.
3. name_jaro_winkler: RapidFuzz Jaro-Winkler similarity between Source 1 name and candidate name.
4. address_token_sort_ratio: RapidFuzz token sort ratio between Source 1 address and candidate address.
5. address_jaro_winkler: RapidFuzz Jaro-Winkler similarity between Source 1 address and candidate address.
6. postal_code_exact: 1 if both records have a postal code and they are exactly equal, else 0.
7. house_number_exact: 1 if both records have a house number and they are exactly equal, else 0.
8. same_first_name_token: 1 if the first whitespace-delimited name token matches, else 0.
9. shared_token_count: Preserved shared token count from blocking candidate generation.

Data loading priority:
1. Cleaned Parquet files (train_source1_clean.parquet, train_source2_clean.parquet, train_source3_clean.parquet)
2. Raw TSV files (train_source1.tsv, train_source2.tsv, train_source3.tsv) with on-the-fly basic cleaning:
   - unidecode transliteration
   - lowercase normalization
   - punctuation removal
   - whitespace normalization
   - address signal extraction (postal codes and house numbers)
3. Fallback sample registry extracted from project exploration artifacts for local testing
   when large raw training datasets are not locally present.

Memory & CPU safety:
- Filtering to only the entity IDs required by the candidate pairs.
- Vectorized and optimized RapidFuzz C++ similarity operations.
- Zero NaN guarantees on all resulting columns.
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import unidecode

# Verify and report RapidFuzz dependency for similarity metrics
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
except ImportError as err:
    sys.exit(
        f"Critical Error: RapidFuzz is required for feature engineering ({err}). "
        f"Please install it using: pip install rapidfuzz==3.11.0"
    )

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Built-in sample registry for the 36 entities used in candidate_pairs_fake_sample.parquet
# Used when full 1.5 GB dataset files are not present in local development
SAMPLE_ENTITY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "S1-777210964": {
        "entity_id": "S1-777210964",
        "business_name": "Davis Family Office",
        "business_address": "88 Olive Circle, Lebanon, TN",
        "country": "US",
    },
    "S2-12029274": {
        "entity_id": "S2-12029274",
        "business_name": "Davis Family Offie",
        "business_address": "88 OLIVE CIR, LEBANON, TN",
        "country": "US",
    },
    "S2-407524980": {
        "entity_id": "S2-407524980",
        "business_name": "DAVIS FAMILY OFFICE",
        "business_address": "OLIVE CIR, LEBANON, TN",
        "country": "US",
    },
    "S3-970714334": {
        "entity_id": "S3-970714334",
        "business_name": "Davis Family (Office)",
        "business_address": "88 Olive Cir, Lebanon, Tennessee",
        "country": "US",
    },
    "S3-712922821": {
        "entity_id": "S3-712922821",
        "business_name": "Davis Family",
        "business_address": "Lebanon, Tennessee, 88 Olive Cir",
        "country": "US",
    },
    "S2-764573417": {
        "entity_id": "S2-764573417",
        "business_name": "-- Holloway Peak Inc Seafood",
        "business_address": "105 ELM ST, MORGANTON, NC",
        "country": "US",
    },
    "S3-202863386": {
        "entity_id": "S3-202863386",
        "business_name": "wilfordhancock.com",
        "business_address": "Mack Rd, Haltom City, Texas",
        "country": "US",
    },
    "S1-190133982": {
        "entity_id": "S1-190133982",
        "business_name": "MS Consultancy Corp",
        "business_address": "Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli, Pune, Maharashtra",
        "country": "India",
    },
    "S2-523219173": {
        "entity_id": "S2-523219173",
        "business_name": "MS [Consultancy]",
        "business_address": "SHYMALA APPTS 1-1ST FLOOR FLAT NO. 4 OPP RATNA HOSPITAL SB ROAD IN HAVELI, PUNE, Maharashtra",
        "country": "India",
    },
    "S3-716445195": {
        "entity_id": "S3-716445195",
        "business_name": "M5  Consultancy Corp",
        "business_address": "Pune, MH, Pune, Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli",
        "country": "India",
    },
    "S3-986804760": {
        "entity_id": "S3-986804760",
        "business_name": "MS Corp Services",
        "business_address": "Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli, Pune, MH",
        "country": "India",
    },
    "S2-166376419": {
        "entity_id": "S2-166376419",
        "business_name": "Ram Marketing Private Limited",
        "business_address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "country": "India",
    },
    "S3-960981775": {
        "entity_id": "S3-960981775",
        "business_name": "Pvt. EFS Print Ventures Ltd.",
        "business_address": "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, Karnataka",
        "country": "India",
    },
    "S1-970131671": {
        "entity_id": "S1-970131671",
        "business_name": "3520 Main Road Realty Inc",
        "business_address": "TX, 158 Simpson Lane, Somerset",
        "country": "US",
    },
    "S2-670035601": {
        "entity_id": "S2-670035601",
        "business_name": "3520 MAIN ROAD REALTY INC",
        "business_address": "SIMPSON LANE, SOMERSET, TX",
        "country": "US",
    },
    "S3-392756625": {
        "entity_id": "S3-392756625",
        "business_name": "3520 MAIN ROAD Realty INC",
        "business_address": "158 Simpson Lane, Somerset, Texas",
        "country": "US",
    },
    "S2-163963287": {
        "entity_id": "S2-163963287",
        "business_name": "Summit Inc",
        "business_address": "GREENSBORO, NC, 19 1/2 STARDUST TRAIL",
        "country": "US",
    },
    "S3-578159284": {
        "entity_id": "S3-578159284",
        "business_name": "LLC Hernandez Colonial Redwood",
        "business_address": "2260- Housecreek Trail, Unit 407, Raleigh, North Carolina",
        "country": "US",
    },
    "S1-391999301": {
        "entity_id": "S1-391999301",
        "business_name": "Team Air Pvt. Ltd.",
        "business_address": "Flat No:101, Anuska Towers, Opp. Mercedes Benz Show Room, Lakdi- Ka, -Pool, Hyderabad, Telangana",
        "country": "India",
    },
    "S2-742819060": {
        "entity_id": "S2-742819060",
        "business_name": "TEAMAIR.COM",
        "business_address": "FLAT NO:101, ANUSKA TOWERS, OPP. MERCEDES BENZ SHOW ROOM, LAKDI- KA, -POOL, Telangana",
        "country": "India",
    },
    "S2-339356009": {
        "entity_id": "S2-339356009",
        "business_name": "Thg Air Pvt. Ltd.",
        "business_address": "FLAT NO:101, -POOL, HYDERABAD, Telangana",
        "country": "India",
    },
    "S2-233734305": {
        "entity_id": "S2-233734305",
        "business_name": "teamair.com",
        "business_address": "FLAT NO:101, ANUSKA TOWERS, OPP. MERCEDES BENZ SHOW ROOM, LAKDI- KA, -POOL, Telangana",
        "country": "India",
    },
    "S3-938838849": {
        "entity_id": "S3-938838849",
        "business_name": "Mirasol",
        "business_address": "Flat No:1-101, Anuska Towers, Opp. Mercedes Benz Show Room, Lakdi- Ka, Hyderabad, -Pool, Andhra Pradesh",
        "country": "India",
    },
    "S3-99519749": {
        "entity_id": "S3-99519749",
        "business_name": "Team Air Pvt. Limited",
        "business_address": "Flat No:1-101, Anuska Towers, Opp. Mercedes Benz Show Room, Lakdi- Ka, -Pool, Hyderabad, Andhra Pradesh",
        "country": "India",
    },
    "S2-639257739": {
        "entity_id": "S2-639257739",
        "business_name": "Aditya Properties LLP",
        "business_address": "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh",
        "country": "India",
    },
    "S1-353020204": {
        "entity_id": "S1-353020204",
        "business_name": "Prairie Capital Partners PLLC",
        "business_address": "120 Autumn Woods Boulevard, Mount Holly, NC",
        "country": "US",
    },
    "S2-403125820": {
        "entity_id": "S2-403125820",
        "business_name": "PRAIRIE CAPITAL PLLC-PARTNERS",
        "business_address": "120. Autumn Woods Boulevard, MOUNT HOLLY, NC",
        "country": "US",
    },
    "S2-482586584": {
        "entity_id": "S2-482586584",
        "business_name": "Prairie (Capital)",
        "business_address": "AUTUMN WOODS BLVD, MOUNT HOLLY, NC",
        "country": "US",
    },
    "S3-740655592": {
        "entity_id": "S3-740655592",
        "business_name": "Prairie Partners PLLC  Services",
        "business_address": "Autumn Woods Blvd, Mount Holly, North Carolina",
        "country": "US",
    },
    "S3-701336150": {
        "entity_id": "S3-701336150",
        "business_name": "prairie capital partners pllc",
        "business_address": "",
        "country": "US",
    },
    "S2-721031885": {
        "entity_id": "S2-721031885",
        "business_name": "Chavira Platinum Chimera LLC",
        "business_address": "282 SAXONY DRIVE, FTT MITCHELL, KY",
        "country": "US",
    },
    "S1-83174673": {
        "entity_id": "S1-83174673",
        "business_name": "Supreme It Private Limited",
        "business_address": "Office No S 07 82Haware, Centurion Plt88-91 Sec19A, Thane, Maharashtra",
        "country": "India",
    },
    "S2-348030512": {
        "entity_id": "S2-348030512",
        "business_name": "Supreme  It Private Limited",
        "business_address": "OFFICE NO S 07 82HAWARE, CENTURION PLT88-91 SEC19A, THANE, Maharashtra",
        "country": "India",
    },
    "S3-755114994": {
        "entity_id": "S3-755114994",
        "business_name": "Supreme IT Private Limited",
        "business_address": "Offiec No S ##07 82Haware, Centurion Plt88-91 Sec19a, Thane, MH",
        "country": "India",
    },
    "S3-444196916": {
        "entity_id": "S3-444196916",
        "business_name": "Supreme It Private",
        "business_address": "Office No S ##07 82Haware, Centurion Plt88-91 Sec19a, Thane, MH",
        "country": "India",
    },
    "S2-508602797": {
        "entity_id": "S2-508602797",
        "business_name": "FOUNDATION EXCEL AGENCY PRIVATE  LIMITED",
        "business_address": "HN 753 E-1, BHARAT NAGAR, 104/1/1 ERANDWANE, Maharashtra",
        "country": "India",
    },
}


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


def clean_text(text: Any) -> str:
    """
    Cleans text by:
    - handling non-string/null inputs safely
    - stripping accents and transliterating with unidecode
    - converting to lowercase
    - replacing punctuation with spaces
    - collapsing multiple consecutive whitespace characters
    """
    if text is None or pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Transliterate / strip accents
    text = unidecode.unidecode(text).lower()

    # Replace punctuation (non-alphanumeric and non-whitespace) with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse consecutive whitespace characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_address_signals(address: Any) -> Dict[str, Optional[str]]:
    """
    Extracts structured invariant signals from an address string:
    - postal_code: standalone 5 or 6 digit sequence (US ZIP or Indian PIN code)
    - house_number: first numerical token in address
    """
    signals: Dict[str, Optional[str]] = {
        "postal_code": None,
        "house_number": None,
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

    return signals


def parse_exploration_output_fallback() -> Dict[str, Dict[str, Any]]:
    """
    Parses sample entities from exploration_output.txt if available,
    supplementing the embedded sample registry.
    """
    output_path = resolve_path("exploration_output.txt")
    if not output_path.exists():
        return {}

    parsed: Dict[str, Dict[str, Any]] = {}
    try:
        with open(output_path, "r", encoding="utf-16le", errors="replace") as f:
            lines = f.readlines()

        current_country = None
        for line in lines:
            if "SAMPLE " in line and "Country:" in line:
                current_country = line.split("Country:")[-1].strip()
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4 and any(parts[1].startswith(pfx) for pfx in ("S1-", "S2-", "S3-")):
                    eid = parts[1]
                    parsed[eid] = {
                        "entity_id": eid,
                        "business_name": parts[2],
                        "business_address": parts[3],
                        "country": current_country,
                    }

        for line in lines:
            m = re.match(r"^\s*\d+\s+(S[23]-[\w\-]+)\s+(.+?)\s{2,}(.+?)\s{2,}(US|India)\s*$", line)
            if m:
                eid, name, addr, country = m.groups()
                if eid not in parsed:
                    parsed[eid] = {
                        "entity_id": eid,
                        "business_name": name.strip(),
                        "business_address": addr.strip(),
                        "country": country.strip(),
                    }
    except Exception:
        pass
    return parsed


def load_source_records(
    source_num: int,
    needed_ids: Optional[Set[str]] = None,
    dataset_type: str = "train",
) -> Dict[str, Dict[str, Any]]:
    """
    Loads entity records for Source 1, 2, or 3 following the roadmap data loading strategy:
    1. First looks for cleaned Parquet files (dataset_processed/{dataset_type}_source{N}_clean.parquet)
    2. Falls back to raw training TSVs (dataset/{dataset_type}/{dataset_type}_source{N}.tsv) and performs basic cleaning
    3. Falls back to sample registry / exploration artifacts for local candidate testing
    """
    label = f"Source {source_num}"
    records: Dict[str, Dict[str, Any]] = {}

    # 1. Check for cleaned Parquet file
    clean_parquet_candidates = [
        f"dataset_processed/{dataset_type}_source{source_num}_clean.parquet",
    ]
    clean_path: Optional[Path] = None
    for cand in clean_parquet_candidates:
        p = resolve_path(cand)
        if p.exists():
            clean_path = p
            break

    if clean_path and clean_path.exists():
        print(f"  [Found Cleaned Parquet] {clean_path.name} for {label}")
        # Define columns of interest
        candidate_cols = [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "cleaned_name",
            "cleaned_address",
            "postal_code",
            "house_number",
        ]
        # Inspect available columns
        schema = pq.read_schema(str(clean_path))
        available_cols = [c for c in candidate_cols if c in schema.names]

        # Read only required rows if needed_ids is reasonably small, or read table
        if needed_ids and len(needed_ids) < 100_000:
            df = pq.read_table(
                str(clean_path),
                columns=available_cols,
                filters=[("entity_id", "in", list(needed_ids))],
            ).to_pandas()
        else:
            df = pd.read_parquet(clean_path, columns=available_cols)

        for _, row in df.iterrows():
            eid = str(row["entity_id"])
            raw_name = row.get("business_name", "")
            raw_addr = row.get("business_address", "")
            c_name = row.get("cleaned_name")
            if pd.isna(c_name) or not c_name:
                c_name = clean_text(raw_name)
            c_addr = row.get("cleaned_address")
            if pd.isna(c_addr) or not c_addr:
                c_addr = clean_text(raw_addr)

            p_code = row.get("postal_code")
            h_num = row.get("house_number")
            if pd.isna(p_code) or not p_code:
                sig = extract_address_signals(raw_addr)
                p_code = sig["postal_code"]
            if pd.isna(h_num) or not h_num:
                sig = extract_address_signals(raw_addr)
                h_num = sig["house_number"]

            records[eid] = {
                "entity_id": eid,
                "business_name": raw_name,
                "business_address": raw_addr,
                "cleaned_name": c_name,
                "cleaned_address": c_addr,
                "postal_code": p_code,
                "house_number": h_num,
                "country": row.get("country", ""),
            }
        print(f"    Loaded {len(records):,} records from cleaned parquet for {label}.")
        return records

    # 2. Fall back to raw TSV files
    raw_tsv_candidates = [
        f"dataset/train/train_source{source_num}.tsv",
        f"dataset/test/test_source{source_num}.tsv",
        f"train_source{source_num}.tsv",
        f"dataset_processed/train_source{source_num}.tsv",
    ]
    raw_path: Optional[Path] = None
    for cand in raw_tsv_candidates:
        p = resolve_path(cand)
        if p.exists():
            raw_path = p
            break

    if raw_path and raw_path.exists():
        print(f"  [Found Raw TSV] {raw_path.name} for {label} (Performing basic cleaning)...")
        chunk_size = 250_000
        for chunk in pd.read_csv(
            raw_path,
            sep="\t",
            chunksize=chunk_size,
            dtype=str,
            keep_default_na=False,
        ):
            if needed_ids:
                chunk = chunk[chunk["entity_id"].isin(needed_ids)]
                if chunk.empty:
                    continue

            for _, row in chunk.iterrows():
                eid = str(row["entity_id"])
                raw_name = row.get("business_name", "")
                raw_addr = row.get("business_address", "")
                c_name = clean_text(raw_name)
                c_addr = clean_text(raw_addr)
                sig = extract_address_signals(raw_addr)

                records[eid] = {
                    "entity_id": eid,
                    "business_name": raw_name,
                    "business_address": raw_addr,
                    "cleaned_name": c_name,
                    "cleaned_address": c_addr,
                    "postal_code": sig["postal_code"],
                    "house_number": sig["house_number"],
                    "country": row.get("country", ""),
                }
            if needed_ids and len(records) >= len(needed_ids):
                break
        print(f"    Loaded and cleaned {len(records):,} records from raw TSV for {label}.")
        return records

    # 3. Fall back to sample registry / exploration data
    print(f"  [Fallback Sample Registry] Cleaned/raw files not found for {label}. Searching sample registry...")
    extra_parsed = parse_exploration_output_fallback()
    combined_sample = {**extra_parsed, **SAMPLE_ENTITY_REGISTRY}

    for eid, data in combined_sample.items():
        if source_num == 1 and not eid.startswith("S1-"):
            continue
        if source_num == 2 and not eid.startswith("S2-"):
            continue
        if source_num == 3 and not eid.startswith("S3-"):
            continue

        if needed_ids and eid not in needed_ids:
            continue

        raw_name = data.get("business_name", "")
        raw_addr = data.get("business_address", "")
        records[eid] = {
            "entity_id": eid,
            "business_name": raw_name,
            "business_address": raw_addr,
            "cleaned_name": clean_text(raw_name),
            "cleaned_address": clean_text(raw_addr),
            "postal_code": extract_address_signals(raw_addr)["postal_code"],
            "house_number": extract_address_signals(raw_addr)["house_number"],
            "country": data.get("country", ""),
        }

    print(f"    Loaded {len(records):,} records from fallback registry for {label}.")
    return records


def calc_token_sort_ratio(s1: str, s2: str) -> float:
    """Computes RapidFuzz token sort ratio. Returns 0.0 if either string is empty."""
    if not s1 or not s2:
        return 0.0
    return float(fuzz.token_sort_ratio(s1, s2))


def calc_token_set_ratio(s1: str, s2: str) -> float:
    """Computes RapidFuzz token set ratio. Returns 0.0 if either string is empty."""
    if not s1 or not s2:
        return 0.0
    return float(fuzz.token_set_ratio(s1, s2))


def calc_jaro_winkler(s1: str, s2: str) -> float:
    """Computes RapidFuzz Jaro-Winkler similarity. Returns 0.0 if either string is empty."""
    if not s1 or not s2:
        return 0.0
    return float(JaroWinkler.similarity(s1, s2))


def calc_exact_match(v1: Optional[str], v2: Optional[str]) -> int:
    """
    Returns 1 if both values are non-empty and exactly equal.
    Returns 0 if either value is None, empty, or nan.
    """
    if v1 is None or v2 is None:
        return 0
    if pd.isna(v1) or pd.isna(v2):
        return 0
    s1, s2 = str(v1).strip(), str(v2).strip()
    if not s1 or not s2 or s1.lower() in ("none", "nan") or s2.lower() in ("none", "nan"):
        return 0
    return 1 if s1 == s2 else 0


def calc_same_first_token(name1: Optional[str], name2: Optional[str]) -> int:
    """
    Returns 1 if the first whitespace-delimited name token matches between the two names.
    Returns 0 otherwise.
    """
    if not name1 or not name2 or pd.isna(name1) or pd.isna(name2):
        return 0
    toks1 = str(name1).split()
    toks2 = str(name2).split()
    if not toks1 or not toks2:
        return 0
    return 1 if toks1[0].lower() == toks2[0].lower() else 0


def generate_candidate_features(
    df_candidates: pd.DataFrame,
    entity_lookup: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Calculates the 9 roadmap matching features for every candidate pair:
    1. name_token_sort_ratio
    2. name_token_set_ratio
    3. name_jaro_winkler
    4. address_token_sort_ratio
    5. address_jaro_winkler
    6. postal_code_exact
    7. house_number_exact
    8. same_first_name_token
    9. shared_token_count (preserved)

    Guarantees no NaN values in the final feature DataFrame.
    """
    n_pairs = len(df_candidates)
    features: List[Dict[str, Any]] = []

    for idx, row in df_candidates.iterrows():
        s1_id = str(row["source1_entity_id"])
        cand_id = str(row["candidate_entity_id"])
        country = str(row.get("country", ""))
        shared_tok = row.get("shared_token_count", 0)
        try:
            shared_tok = int(shared_tok) if not pd.isna(shared_tok) else 0
        except (ValueError, TypeError):
            shared_tok = 0

        s1_rec = entity_lookup.get(s1_id, {})
        c_rec = entity_lookup.get(cand_id, {})

        s1_name = s1_rec.get("cleaned_name", "")
        c_name = c_rec.get("cleaned_name", "")
        s1_addr = s1_rec.get("cleaned_address", "")
        c_addr = c_rec.get("cleaned_address", "")

        # 1. name_token_sort_ratio
        name_sort = calc_token_sort_ratio(s1_name, c_name)

        # 2. name_token_set_ratio
        name_set = calc_token_set_ratio(s1_name, c_name)

        # 3. name_jaro_winkler
        name_jw = calc_jaro_winkler(s1_name, c_name)

        # 4. address_token_sort_ratio
        addr_sort = calc_token_sort_ratio(s1_addr, c_addr)

        # 5. address_jaro_winkler
        addr_jw = calc_jaro_winkler(s1_addr, c_addr)

        # 6. postal_code_exact
        pc_exact = calc_exact_match(s1_rec.get("postal_code"), c_rec.get("postal_code"))

        # 7. house_number_exact
        hn_exact = calc_exact_match(s1_rec.get("house_number"), c_rec.get("house_number"))

        # 8. same_first_name_token
        first_tok_match = calc_same_first_token(s1_name, c_name)

        # Country fallback if candidate row had empty country
        if not country or country == "nan":
            country = s1_rec.get("country") or c_rec.get("country") or "Unknown"

        features.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "country": country,
            "name_token_sort_ratio": name_sort,
            "name_token_set_ratio": name_set,
            "name_jaro_winkler": name_jw,
            "address_token_sort_ratio": addr_sort,
            "address_jaro_winkler": addr_jw,
            "postal_code_exact": pc_exact,
            "house_number_exact": hn_exact,
            "same_first_name_token": first_tok_match,
            "shared_token_count": shared_tok,
        })

    df_feat = pd.DataFrame(features)

    # Enforce exact column order and explicit types
    cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "country",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_jaro_winkler",
        "address_token_sort_ratio",
        "address_jaro_winkler",
        "postal_code_exact",
        "house_number_exact",
        "same_first_name_token",
        "shared_token_count",
    ]
    df_feat = df_feat[cols]

    # Explicit NaN check and fill
    for c in ["name_token_sort_ratio", "name_token_set_ratio", "name_jaro_winkler", "address_token_sort_ratio", "address_jaro_winkler"]:
        df_feat[c] = df_feat[c].fillna(0.0).astype(float)
    for c in ["postal_code_exact", "house_number_exact", "same_first_name_token", "shared_token_count"]:
        df_feat[c] = df_feat[c].fillna(0).astype("int32")
    for c in ["source1_entity_id", "candidate_entity_id", "country"]:
        df_feat[c] = df_feat[c].fillna("").astype(str)

    return df_feat


def run_feature_pipeline(
    candidate_path_str: str,
    output_path_str: Optional[str] = None,
    dataset_type: str = "train",
) -> Path:
    """
    Executes the end-to-end feature extraction pipeline:
    1. Loads candidate pairs.
    2. Identifies required entity IDs.
    3. Loads corresponding Source 1, 2, and 3 records.
    4. Computes all 9 matching features.
    5. Saves resulting feature table to Parquet.
    6. Displays execution summary and verification stats.
    """
    t_start = time.time()
    cand_path = resolve_path(candidate_path_str)
    if not cand_path.exists():
        sys.exit(f"Error: Candidate file not found at: {cand_path}")

    # Determine output path
    if output_path_str:
        out_path = Path(output_path_str)
        if not out_path.is_absolute():
            out_path = resolve_path(output_path_str)
    else:
        # Default behavior:
        # If candidate is candidate_pairs_fake_sample.parquet -> features_fake_sample.parquet
        output_dir = resolve_path("dataset_processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        if "fake_sample" in cand_path.name:
            out_path = output_dir / "features_fake_sample.parquet"
        else:
            out_path = output_dir / f"features_{cand_path.stem}.parquet"

    print("=" * 80)
    print("      BUSINESS ENTITY RESOLUTION - FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    print(f"Candidates Input Path : {cand_path}")
    print(f"Output Parquet Path   : {out_path}")
    print(f"String Similarity Lib : RapidFuzz (Jaro-Winkler: rapidfuzz.distance.JaroWinkler)")

    # 1. Load candidate pairs
    df_candidates = pd.read_parquet(cand_path)
    n_candidates = len(df_candidates)
    print(f"\nLoaded {n_candidates:,} candidate pairs from {cand_path.name}")

    # 2. Extract needed entity IDs
    s1_ids = set(df_candidates["source1_entity_id"].dropna().astype(str))
    cand_ids = set(df_candidates["candidate_entity_id"].dropna().astype(str))
    s2_ids = {cid for cid in cand_ids if cid.startswith("S2-")}
    s3_ids = {cid for cid in cand_ids if cid.startswith("S3-")}
    other_ids = cand_ids - s2_ids - s3_ids

    print(f"Unique Source 1 Entities : {len(s1_ids):,}")
    print(f"Unique Candidate Entities: {len(cand_ids):,} (Source 2: {len(s2_ids):,}, Source 3: {len(s3_ids):,})")

    # 3. Load entity records across sources
    print("\nLoading entity records across sources...")
    s1_records = load_source_records(1, needed_ids=s1_ids, dataset_type=dataset_type)
    s2_records = load_source_records(2, needed_ids=(s2_ids | other_ids), dataset_type=dataset_type)
    s3_records = load_source_records(3, needed_ids=(s3_ids | other_ids), dataset_type=dataset_type)

    # Combine into a single fast lookup
    entity_lookup: Dict[str, Dict[str, Any]] = {}
    entity_lookup.update(s1_records)
    entity_lookup.update(s2_records)
    entity_lookup.update(s3_records)

    total_loaded = len(entity_lookup)
    all_needed = s1_ids | cand_ids
    found_count = sum(1 for eid in all_needed if eid in entity_lookup)
    print(f"Total entity records loaded: {total_loaded:,} (Coverage: {found_count}/{len(all_needed)} required entities)")

    # 4. Generate features
    print(f"\nCalculating features for {n_candidates:,} candidate pairs...")
    t_feat_start = time.time()
    df_features = generate_candidate_features(df_candidates, entity_lookup)
    t_feat_end = time.time()
    print(f"Feature extraction completed in {t_feat_end - t_feat_start:.2f}s "
          f"({n_candidates / max(t_feat_end - t_feat_start, 0.001):.0f} pairs/s)")

    # 5. Save output Parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_features, preserve_index=False)
    pq.write_table(table, str(out_path), compression="snappy")
    print(f"\nFeatures saved successfully to: {out_path} ({out_path.stat().st_size / 1024:.2f} KB)")

    # 6. Verification and Reporting as requested
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("                    FEATURE EXTRACTION REPORT")
    print("=" * 80)
    print(f"1. Number of candidate pairs processed : {n_candidates:,}")
    print(f"2. Number of feature rows produced    : {len(df_features):,}")
    print(f"3. Complete list of feature columns ({len(df_features.columns)} columns):")
    for col in df_features.columns:
        print(f"   - {col} (dtype: {df_features[col].dtype})")

    print(f"\n4. First 10 rows of the resulting feature table:")
    print(df_features.head(10).to_string())

    has_nans = df_features.isna().any().any()
    print(f"\n5. Any columns contain NaN values: {has_nans}")
    if has_nans:
        nan_summary = df_features.isna().sum().to_dict()
        print(f"   Warning! NaN counts per column: {nan_summary}")
    else:
        print("   [VERIFIED] All columns are completely free of NaN values.")

    print(f"\nTotal Pipeline Execution Time: {total_time:.2f}s")
    print("=" * 80)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline for Business Entity Resolution"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="dataset_processed/candidate_pairs_fake_sample.parquet",
        help="Path to candidate pairs parquet file (default: dataset_processed/candidate_pairs_fake_sample.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output features parquet file (default: dataset_processed/features_fake_sample.parquet)",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset type (train or test)",
    )

    args = parser.parse_args()
    run_feature_pipeline(
        candidate_path_str=args.candidates,
        output_path_str=args.output,
        dataset_type=args.dataset_type,
    )


if __name__ == "__main__":
    main()
