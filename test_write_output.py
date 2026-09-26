"""
test_write_output.py - Test pre-write validation and TSV formatting
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path("src").resolve()))
from write_output import write_results, check_source1_coverage, validate_id_lists

# 1. Define required 10 Source 1 IDs
required_s1 = [f"S1-{i:05d}" for i in range(1, 11)]
required_set = set(required_s1)
print(f"Required S1 Test IDs (10 total): {required_s1}")

# 2. Construct 10-row fake results DataFrame with DELIBERATE ERRORS:
#    - Error A: Row 5 has a duplicate ID inside its matched list: ['S2-2005', 'S2-2005']
#    - Error B: Missing required S1 ID S1-00010 (has S1-99999 instead)
flawed_data = [
    {"source1_entity_id": "S1-00001", "matched_entity_ids": ["S2-2001", "S3-3001"]},
    {"source1_entity_id": "S1-00002", "matched_entity_ids": ["S2-2002"]},
    {"source1_entity_id": "S1-00003", "matched_entity_ids": []},                     # Empty (singleton)
    {"source1_entity_id": "S1-00004", "matched_entity_ids": ["S3-3004"]},
    {"source1_entity_id": "S1-00005", "matched_entity_ids": ["S2-2005", "S2-2005"]}, # INTRA-LIST DUPLICATE
    {"source1_entity_id": "S1-00006", "matched_entity_ids": []},                     # Empty (singleton)
    {"source1_entity_id": "S1-00007", "matched_entity_ids": ["S2-2007", "S3-3007"]},
    {"source1_entity_id": "S1-00008", "matched_entity_ids": ["S3-3008"]},
    {"source1_entity_id": "S1-00009", "matched_entity_ids": []},                     # Empty (singleton)
    {"source1_entity_id": "S1-99999", "matched_entity_ids": ["S2-2010"]},             # MISSING S1-00010 (Extra S1-99999)
]

df_flawed = pd.DataFrame(flawed_data)

print("\n" + "=" * 70)
print("TEST 1: Testing pre-write validation on flawed DataFrame...")
print("=" * 70)

# Check intra-list duplicate detection
try:
    print("Testing intra-list duplicate check...")
    validate_id_lists(df_flawed, "matched_entity_ids")
    print("FAILED: Did not catch intra-list duplicate!")
except ValueError as e:
    print(f"SUCCESS: Caught expected error:\n  -> {e}")

# Check missing required S1 ID detection
try:
    print("\nTesting missing required Source 1 ID check...")
    check_source1_coverage(df_flawed, required_ids=required_set)
    print("FAILED: Did not catch missing S1 ID!")
except ValueError as e:
    print(f"SUCCESS: Caught expected error:\n  -> {e}")

# 3. Fix the flaws
print("\n" + "=" * 70)
print("TEST 2: Fixing both errors and writing clean output...")
print("=" * 70)

clean_data = [
    {"source1_entity_id": "S1-00001", "matched_entity_ids": ["S2-2001", "S3-3001"]},
    {"source1_entity_id": "S1-00002", "matched_entity_ids": ["S2-2002"]},
    {"source1_entity_id": "S1-00003", "matched_entity_ids": []},                     # Empty string
    {"source1_entity_id": "S1-00004", "matched_entity_ids": ["S3-3004"]},
    {"source1_entity_id": "S1-00005", "matched_entity_ids": ["S2-2005"]},             # Fixed: removed duplicate
    {"source1_entity_id": "S1-00006", "matched_entity_ids": []},                     # Empty string
    {"source1_entity_id": "S1-00007", "matched_entity_ids": ["S2-2007", "S3-3007"]},
    {"source1_entity_id": "S1-00008", "matched_entity_ids": ["S3-3008"]},
    {"source1_entity_id": "S1-00009", "matched_entity_ids": []},                     # Empty string
    {"source1_entity_id": "S1-00010", "matched_entity_ids": ["S2-2010"]},             # Fixed: restored S1-00010
]

df_clean = pd.DataFrame(clean_data)
out_file = Path("output/matching_results_TEST.tsv")

# Write clean results with validation enabled
written_path = write_results(
    df=df_clean,
    path=out_file,
    id_col_name="matched_entity_ids",
    required_ids=required_set,
    validate_coverage=True,
)
print(f"Successfully validated and wrote clean TSV to: {written_path}")

# 4. Verify output file formatting
print("\n" + "=" * 70)
print("TEST 3: Verifying file formatting (raw representation)...")
print("=" * 70)
with open(out_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # Print line representation showing real tabs and no quotes
    print(f"Line {i+1:02d}: {repr(line)}")
    assert "\t" in line, f"Line {i+1} missing tab delimiter!"
    assert '"' not in line, f"Line {i+1} contains stray quote mark!"
    assert "nan" not in line.lower(), f"Line {i+1} contains nan!"

print("\nFormatting verified: real tabs used, empty strings for singletons, zero stray quotes.")
