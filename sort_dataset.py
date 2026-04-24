import os
import shutil
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────
CSV_PATH   = r"C:\Users\xiaon\Downloads\Training+Testing_data+label\Training+Testing_data_label.csv"
IMG_DIR    = r"C:\Users\xiaon\Downloads\Training+Testing_data+label\Training+Testing_data"
OUTPUT_DIR = r"C:\Users\xiaon\Downloads\dr ai\project\dataset\train"

# ── Class mapping ────────────────────────────────────────────────────
CLASS_MAP = {
    0: "0_No_DR",
    1: "1_Mild",
    2: "2_Moderate",
    3: "3_Severe",
    4: "4_PDR"
}

# ── Create output folders ────────────────────────────────────────────
for name in CLASS_MAP.values():
    os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)
print("✅ Created 5 class folders")

# ── Read labels CSV ──────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded CSV — {len(df)} images found")
print(f"   Columns: {list(df.columns)}")
print(f"   Sample:\n{df.head()}\n")

# ── Debug: check what's actually in the image folder ─────────────────
actual_files = set(os.listdir(IMG_DIR))
csv_files_raw = set(df['Image'].astype(str))

# Normalize: add .png if missing extension
csv_files_normalized = set(
    f if f.lower().endswith(('.png', '.jpg', '.jpeg')) else f + '.png'
    for f in csv_files_raw
)

in_csv_not_in_folder = csv_files_normalized - actual_files
in_folder_not_in_csv = actual_files - csv_files_normalized

print("=" * 55)
print("  DIAGNOSTIC REPORT")
print("=" * 55)
print(f"  Total rows in CSV              : {len(df)}")
print(f"  Total files in image folder    : {len(actual_files)}")
print(f"  Files in CSV but NOT in folder : {len(in_csv_not_in_folder)}")
print(f"  Files in folder but NOT in CSV : {len(in_folder_not_in_csv)}")

if in_csv_not_in_folder:
    print(f"\n  Sample missing files (first 10):")
    for f in sorted(list(in_csv_not_in_folder))[:10]:
        print(f"    ✗ {f}")

if in_folder_not_in_csv:
    print(f"\n  Sample unmatched folder files (first 10):")
    for f in sorted(list(in_folder_not_in_csv))[:10]:
        print(f"    ? {f}")

# Check for subfolders inside IMG_DIR
subfolders = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
if subfolders:
    print(f"\n  ⚠️  Subfolders detected inside image folder:")
    for sf in subfolders:
        sf_path = os.path.join(IMG_DIR, sf)
        sf_count = len(os.listdir(sf_path))
        print(f"    📁 {sf}/ — {sf_count} files")
    print("     → Images may be split across subfolders!")

print("=" * 55 + "\n")

# ── Sort images into folders ─────────────────────────────────────────
moved   = 0
missing = 0
missing_files = []

for _, row in df.iterrows():
    img_name = str(row['Image'])
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_name = img_name + '.png'

    src = os.path.join(IMG_DIR, img_name)

    # Try alternate extensions if not found
    if not os.path.exists(src):
        base = os.path.splitext(img_name)[0]
        for ext in ['.png', '.jpeg', '.jpg']:
            candidate = os.path.join(IMG_DIR, base + ext)
            if os.path.exists(candidate):
                src = candidate
                img_name = base + ext
                break

    # Try searching in subfolders if still not found
    if not os.path.exists(src) and subfolders:
        for sf in subfolders:
            for ext in ['.png', '.jpeg', '.jpg']:
                base = os.path.splitext(img_name)[0]
                candidate = os.path.join(IMG_DIR, sf, base + ext)
                if os.path.exists(candidate):
                    src = candidate
                    img_name = base + ext
                    break
            if os.path.exists(src):
                break

    grade = int(row['Label'])
    cls   = CLASS_MAP.get(grade)

    if cls and os.path.exists(src):
        dst = os.path.join(OUTPUT_DIR, cls, os.path.basename(src))
        shutil.copy(src, dst)
        moved += 1
    else:
        missing += 1
        missing_files.append(img_name)

# ── Summary ──────────────────────────────────────────────────────────
print(f"✅ Sorted  : {moved} images")
print(f"⚠️  Missing : {missing} images")

if missing_files:
    print(f"\n  First 10 unresolved missing files:")
    for f in missing_files[:10]:
        print(f"    ✗ {f}")

print(f"\nFinal distribution:")
for name in CLASS_MAP.values():
    count = len(os.listdir(os.path.join(OUTPUT_DIR, name)))
    bar   = "█" * (count // 50)
    print(f"  {name:20s}: {count:5d}  {bar}")

print(f"\n🎉 Done! Images sorted into:\n{OUTPUT_DIR}")