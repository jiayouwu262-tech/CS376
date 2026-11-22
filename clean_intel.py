
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Intel/polite-guard")

for split in dataset:
    df = dataset[split].to_pandas()

    print(f"\n--- {split.upper()} ---")
    print("Rows:", len(df))
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nSource distribution:")
    print(df["source"].value_counts(dropna=False))

    lengths = df["text"].fillna("").astype(str).str.split().str.len()

    print("\nText length statistics:")
    print(lengths.describe())

    print("\nExample rows:")
    print(df[["text", "label", "source"]].head())

dfs = {
    split: dataset[split].to_pandas()
    for split in dataset
}

# Check missing values and empty strings
for split, df in dfs.items():
    print(f"\n--- {split.upper()} ---")

    print("Missing text:", df["text"].isna().sum())

    empty = df["text"].fillna("").str.strip().eq("")
    print("Empty text:", empty.sum())

    print("Duplicate text within split:",
          df["text"].duplicated().sum())

# Check overlap between splits
train_texts = set(dataset["train"]["text"])
validation_texts = set(dataset["validation"]["text"])
test_texts = set(dataset["test"]["text"])

print("\n--- CROSS-SPLIT OVERLAP ---")
print("Train ∩ Validation:", len(train_texts & validation_texts))
print("Train ∩ Test:", len(train_texts & test_texts))
print("Validation ∩ Test:", len(validation_texts & test_texts))

# Inspect the LMS examples
test_df = dataset["test"].to_pandas()
lms = test_df[test_df["source"] == "LMS"]

print("\n--- LMS EXAMPLES ---")
print("Number of LMS examples:", len(lms))
print(lms["label"].value_counts())
print(lms[["text", "label"]].head(10).to_string(index=False))

overlap = train_texts & test_texts

for text in overlap:
    print("\nOverlapping text:", repr(text))

    print("\nTRAIN:")
    print(dfs["train"][dfs["train"]["text"] == text][
        ["label", "source"]
    ])

    print("\nTEST:")
    print(dfs["test"][dfs["test"]["text"] == text][
        ["label", "source"]
    ])


import re
import os

def clean_text(text):
    if pd.isna(text):
        return None

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text if text else None


# Clean each split
cleaned = {}

for split, df in dfs.items():
    df = df.copy()

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].str.strip().str.lower()

    # Remove missing/empty text
    df = df.dropna(subset=["text", "label"])

    # Remove exact duplicates within this split
    df = df.drop_duplicates(subset=["text"])

    cleaned[split] = df.reset_index(drop=True)

# Remove the one cross-split duplicate from training only
train_texts = set(cleaned["train"]["text"])
test_texts = set(cleaned["test"]["text"])

overlap = train_texts & test_texts

cleaned["train"] = cleaned["train"][
    ~cleaned["train"]["text"].isin(overlap)
].reset_index(drop=True)

# Save files
output_dir = os.path.expanduser("~/intel_cleaned")
os.makedirs(output_dir, exist_ok=True)

for split, df in cleaned.items():
    df.to_csv(
    os.path.join(output_dir, f"intel_{split}.csv"),
    index=False
    )

# Save LMS separately for evaluation
lms = cleaned["test"][cleaned["test"]["source"] == "LMS"]

lms.to_csv(
    os.path.join(output_dir, "intel_test_lms.csv"),
    index=False
)

print("\n--- CLEANING SUMMARY ---")

for split, df in cleaned.items():
    print(split, len(df))


# ==========================================
# STEP 6: CREATE MAPPING A AND MAPPING B
# ==========================================

def create_mapping_a(df):
    """
    Mapping A:
    Keep only explicitly polite and impolite examples.
    Discard somewhat polite and neutral.
    """

    result = df[
        df["label"].isin(["polite", "impolite"])
    ].copy()

    result["binary_label"] = result["label"].map({
        "polite": 1,
        "impolite": 0
    })

    return result.reset_index(drop=True)


def create_mapping_b(df):
    """
    Mapping B:
    Polite + somewhat polite = 1
    Neutral + impolite = 0
    """

    result = df.copy()

    result["binary_label"] = result["label"].map({
        "polite": 1,
        "somewhat polite": 1,
        "neutral": 0,
        "impolite": 0
    })

    return result.reset_index(drop=True)


# Apply both mappings to all three splits
intel_a = {
    split: create_mapping_a(df)
    for split, df in cleaned.items()
}

intel_b = {
    split: create_mapping_b(df)
    for split, df in cleaned.items()
}


# ==========================================
# STEP 7: SAVE MAPPED DATASETS
# ==========================================

for split in cleaned:

    intel_a[split].to_csv(
        os.path.join(
            output_dir,
            f"intel_{split}_mapping_a.csv"
        ),
        index=False
    )

    intel_b[split].to_csv(
        os.path.join(
            output_dir,
            f"intel_{split}_mapping_b.csv"
        ),
        index=False
    )


# ==========================================
# STEP 8: VERIFY LABEL DISTRIBUTIONS
# ==========================================

for name, mapping in [("A", intel_a), ("B", intel_b)]:

    print(f"\n========== MAPPING {name} ==========")

    for split, df in mapping.items():

        print(f"\n{split.upper()}")
        print("Total rows:", len(df))

        print(
            df["binary_label"]
            .value_counts()
            .sort_index()
            .rename(index={0: "Impolite (0)", 1: "Polite (1)"})
        )


# ==========================================
# STEP 9: SUBSAMPLE INTEL TRAINING DATA
# ==========================================

# Original paper's dataset sizes
sample_sizes = {
    "wiki_sized": 2178,
    "se_sized": 3302
}

def stratified_sample(df, total_size, random_state=42):
    """
    Create a balanced sample while preserving
    the proportion of each binary class.
    """

    class_size = total_size // 2

    polite = df[df["binary_label"] == 1].sample(
        n=class_size,
        random_state=random_state
    )

    impolite = df[df["binary_label"] == 0].sample(
        n=class_size,
        random_state=random_state
    )

    result = pd.concat([polite, impolite])

    return result.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)


# Create samples for both mappings
for mapping_name, mapping in [("A", intel_a), ("B", intel_b)]:

    train_df = mapping["train"]

    for size_name, total_size in sample_sizes.items():

        sampled = stratified_sample(
            train_df,
            total_size
        )

        filename = (
            f"intel_train_mapping_{mapping_name}_{size_name}.csv"
        )

        sampled.to_csv(
            os.path.join(output_dir, filename),
            index=False
        )

        print(f"\n{filename}")
        print("Total:", len(sampled))
        print(sampled["binary_label"].value_counts())
