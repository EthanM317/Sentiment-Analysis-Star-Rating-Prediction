
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    # Root directory
    ROOT_DIR = Path(".")

    # Output file
    OUTPUT_FILE = ROOT_DIR / "imdb_ratings.csv"

    # Store all dataframes
    dataframes = []

    # Store the original files that will be deleted
    files_to_delete = []

    # Read all CSV files in the root directory
    for file_path in ROOT_DIR.glob("*.csv"):

        # Ignore the output file and IMDB Dataset.csv
        if file_path.name in [OUTPUT_FILE.name, "IMDB-Dataset.csv"]:
            continue

        print(f"Processing {file_path}...")

        df = pd.read_csv(file_path)

        # Keep only the rating and review columns
        df = df[["Rating", "Review"]].copy()

        # Rename columns to lowercase
        df.columns = ["rating", "review"]

        dataframes.append(df)

        # Mark this original file for deletion
        files_to_delete.append(file_path)

    # Ensure there are datasets to combine
    if not dataframes:
        print("No CSV files found to combine.")
        raise SystemExit

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Shuffle the rows reproducibly
    combined_df = combined_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Save the combined dataset
    combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCombined {len(combined_df):,} reviews.")
    print(f"Saved to {OUTPUT_FILE.resolve()}")

    # Delete the original files after successfully saving
    print("\nDeleting original CSV files...")

    for file_path in files_to_delete:
        file_path.unlink()
        print(f"Deleted {file_path}")

    print("\nFinished!")
