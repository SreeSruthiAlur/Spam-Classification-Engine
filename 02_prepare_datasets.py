# 02_prepare_datasets.py
# Loads, cleans, and saves all datasets into a standard schema.
# Output: spamassassin_cleaned.csv, sms_cleaned.csv, pooled_cleaned.csv

import pandas as pd
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
SA_IN       = os.path.join(BASE, "spamassassin", "spamassassin_parsed.csv")
SMS_IN      = os.path.join(BASE, "sms_spam", "SMSSpamCollection")
SA_OUT      = os.path.join(BASE, "spamassassin_cleaned.csv")
SMS_OUT     = os.path.join(BASE, "sms_cleaned.csv")
POOLED_OUT  = os.path.join(BASE, "pooled_cleaned.csv")


def clean_spamassassin():
    print("=== SpamAssassin ===")
    df = pd.read_csv(SA_IN)

    # Drop rows with missing label or empty text
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"].str.strip().str.len() > 0].copy()

    # Ensure label is integer
    df["label"] = df["label"].astype(int)

    # Deduplicate on text
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"  Removed {before - len(df)} duplicates.")

    # Standardise metadata columns
    df["domain"] = df["domain"].fillna("").str.strip().str.lower()
    df["unsub"]  = df["unsub"].fillna(0).astype(int)
    df["date"]   = df["date"].fillna("")
    df["sender"] = df["sender"].fillna("")

    # Add corpus tag
    df["corpus"] = "spamassassin"

    df = df[["label", "text", "sender", "domain", "date", "unsub", "corpus"]]
    df.to_csv(SA_OUT, index=False)

    print(f"  Final shape : {df.shape}")
    print(f"  Labels      : {df['label'].value_counts().to_dict()}")
    print(f"  Unsub flag  : {df['unsub'].value_counts().to_dict()}")
    print(f"  Saved to    : {SA_OUT}\n")
    return df


def clean_sms():
    print("=== SMS Spam Collection ===")
    df = pd.read_csv(
        SMS_IN,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1"
    )

    df["label"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"].str.strip().str.len() > 0].copy()
    df["label"] = df["label"].astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"  Removed {before - len(df)} duplicates.")

    # SMS has no metadata — fill with None so pipeline doesn't break
    df["sender"] = None
    df["domain"] = None
    df["date"]   = None
    df["unsub"]  = None
    df["corpus"] = "sms"

    df = df[["label", "text", "sender", "domain", "date", "unsub", "corpus"]]
    df.to_csv(SMS_OUT, index=False)

    print(f"  Final shape : {df.shape}")
    print(f"  Labels      : {df['label'].value_counts().to_dict()}")
    print(f"  Saved to    : {SMS_OUT}\n")
    return df


def make_pooled(sa_df, sms_df):
    print("=== Pooled (SpamAssassin + SMS) ===")
    df = pd.concat([sa_df, sms_df], ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"  Removed {before - len(df)} cross-corpus duplicates.")

    df.to_csv(POOLED_OUT, index=False)
    print(f"  Final shape : {df.shape}")
    print(f"  Labels      : {df['label'].value_counts().to_dict()}")
    print(f"  Saved to    : {POOLED_OUT}\n")
    return df


if __name__ == "__main__":
    sa_df  = clean_spamassassin()
    sms_df = clean_sms()
    make_pooled(sa_df, sms_df)
    print("All datasets prepared successfully.")