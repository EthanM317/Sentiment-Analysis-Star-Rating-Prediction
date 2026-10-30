
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    # Root directory
    ROOT_DIR = Path(".")

    # Output file
    OUTPUT_FILE = ROOT_DIR / "imdb_ratings.csv"

    # Store all dataframes
    dataframes = []

    # Read all CSV files in the root directory, excluding the output file
    for file_path in ROOT_DIR.glob("*.csv"):
        if file_path.name == OUTPUT_FILE.name:
            continue

        print(f"Processing {file_path}...")

        df = pd.read_csv(file_path)

        # Keep only the rating and review columns, dropping MovieID
        df = df[["Rating", "Review"]].copy()

        # Rename columns to lowercase
        df.columns = ["rating", "review"]

        dataframes.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Shuffle the rows reproducibly
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the combined dataset in the root directory
    combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCombined {len(combined_df):,} reviews.")
    print(f"Saved to {OUTPUT_FILE.resolve()}")
