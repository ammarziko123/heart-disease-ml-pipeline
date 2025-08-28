import os, urllib.request, ssl, sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEST = os.path.join(DATA_DIR, "heart.csv")

MIRRORS = [
    # Common mirrors that host the Kaggle-style 'heart.csv' (no auth required)
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",  # fallback demo (shape differs)
    "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/heart/heart.csv",
    "https://raw.githubusercontent.com/SathwikTejaswi/heart-disease-prediction/master/heart.csv",
    "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv",
]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in MIRRORS:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, DEST)
            print(f"Downloaded to {DEST}")
            # quick sanity check
            import csv
            with open(DEST, newline='', encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
            if "target" in [h.lower() for h in header]:
                print("Found expected 'target' column. Done.")
                return
            else:
                print("Warning: 'target' column not found, this file may be a placeholder/different schema.")
                # keep file but continue trying next mirror
        except Exception as e:
            print(f"Failed: {e}")
    print("Finished attempts. If schema mismatches, please place a correct heart.csv into ./data manually.")

if __name__ == "__main__":
    main()
