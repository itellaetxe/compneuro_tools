import os

from argparse import ArgumentParser

import numpy as np
import polars as pl

TERMINATIONS = {" ": ".txt",
                ",": ".csv",
                "\t": ".tsv",
                ";": ".csv"}

def _setup_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Subsample majority group by matching with minority group, according to a column.")
    parser.add_argument(
        "--dataframe",
        type=str,
        required=True,
        help="Path to the input DataFrame. In ",
    )
    parser.add_argument(
        "--group1",
        type=str,
        required=True,
        help="Column name for the first group (e.g., 'g1_SN'). It has to be present in the DataFrame.",
    )
    parser.add_argument(
        "--group2",
        type=str,
        required=True,
        help="Column name for the second group (e.g., 'g2_PV'). It has to be present in the DataFrame.",
    )
    parser.add_argument(
        "--matching_column",
        type=str,
        required=True,
        help="Column name to use for matching.",
    )
    parser.add_argument(
        "--caliper",
        type=float,
        required=False,
        default=None,
        help="Maximum age difference allowed for matching. Default is None (no caliper).",
    )
    parser.add_argument(
        "--separator",
        type=str,
        required=False,
        default="\t",
        help="Separator used in the input DataFrame. Default is '\\t'.",
    )
    parser.add_argument(
        "--has_header",
        action="store_true",
        required=False,
        help="Indicates if the input DataFrame has a header. Default is True.",)

    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to the output directory. If not provided, the output will be saved in the same directory as the input DataFrame.",
    )
    return parser


def _check_args(args) -> None:
    # Check if the input file exists
    if not os.path.exists(args.dataframe):
        raise FileNotFoundError(f"Input file {args.dataframe} does not exist.")
    else:
        args.dataframe_path = os.path.abspath(args.dataframe)

    # Check if the output path exists, if not create it
    if args.output is not None and not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    elif args.output is not None and not os.path.isdir(args.output):
        raise NotADirectoryError(f"Output path {args.output} is not a directory.")
    else: # if output is None, set it to the same directory as the input file
        args.output = os.path.dirname(args.dataframe)

    # Check if the separator is valid
    if args.separator not in [" ", ",", "\t", ";"]:
        raise ValueError(f"Invalid separator '{args.separator}'. Valid options are ' ', ',', '\\t', ';'.")

    args.dataframe = pl.read_csv(args.dataframe_path,
                                 separator=args.separator,
                                 has_header=args.has_header,
                                 infer_schema=False)

    # Remove whitespaces in dataframe column names
    cols = [col.strip() for col in args.dataframe.columns]
    args.dataframe = args.dataframe.rename({old: new for old, new in zip(args.dataframe.columns, cols)})
    # Check if the group columns are present in the DataFrame
    if args.group1 not in args.dataframe.columns:
        raise ValueError(f"Column '{args.group1}' not found in the DataFrame.")
    if args.group2 not in args.dataframe.columns:
        raise ValueError(f"Column '{args.group2}' not found in the DataFrame.")
    if args.matching_column not in args.dataframe.columns:
        raise ValueError(f"Matching column '{args.matching_column}' not found in the DataFrame.")

    # Make matching column numeric, cast to pl.Float32
    args.dataframe = args.dataframe.with_columns(pl.col(args.matching_column).cast(pl.Float32).alias(args.matching_column))
    return args


def subsample_majority_by_age_match(df: pl.DataFrame,
                                    group1: str,
                                    group2: str,
                                    matching_column: str,
                                    caliper: float = None) -> pl.DataFrame:
    """
    Subsample subjects from the majority group by finding age matches from the minority group.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing subject data
    group1 : str
        Column name for the first group (e.g., "g1_SN")
    group2 : str
        Column name for the second group (e.g., "g2_PV")
    matching_column : str
        Column name to use for matching
    caliper : float, optional
        Maximum age difference allowed for matching (in years)
        
    Returns:
    --------
    matched_df : polars.DataFrame
        DataFrame containing subjects from both groups after matching
    """

    # Extract subjects from each group
    group1_df = df.filter(pl.col(group1) == "1")
    group2_df = df.filter(pl.col(group2) == "1")
    
    # Determine which is minority and majority group
    if len(group1_df) <= len(group2_df):
        minority_df = group1_df.with_row_index()
        majority_df = group2_df.with_row_index()
        minority_name, majority_name = group1, group2
    else:
        minority_df = group2_df.with_row_index()
        majority_df = group1_df.with_row_index()
        minority_name, majority_name = group2, group1
    
    # Lists to store matched indices
    minority_matched_indices = []
    majority_matched_indices = []
    
    # Get ages as numpy arrays for faster processing
    minority_ages = minority_df[matching_column].to_numpy()
    majority_ages = majority_df[matching_column].to_numpy()
    majority_indices = np.arange(len(majority_ages))

    mean_diff = np.abs(minority_df[matching_column].mean() - majority_df[matching_column].mean())
    print(f"Original sizes -> {minority_name} (minority): {len(minority_df)}, {majority_name} (majority): {len(majority_df)}"
          f"\nMean absolute difference before matching: {mean_diff:.3f}\n")
    
    # For each subject in minority group, find closest match in majority group
    for i, minority_age in enumerate(minority_ages):
        age_diffs = np.abs(majority_ages - minority_age)
        
        # If using a caliper, skip if no matches within caliper
        if caliper is not None and np.min(age_diffs) > caliper:
            continue
        elif np.min(age_diffs) > np.std(minority_ages):
            print(f"Adding subject with age difference higher than minority "
                  f"group age standard deviation ({np.min(age_diffs):.3f} > {np.std(minority_ages):.3f})")

        # Find the index of the best match
        best_match_idx = np.argmin(age_diffs)
        
        minority_matched_indices.append(i)
        majority_matched_indices.append(majority_indices[best_match_idx])
        
        # Remove the matched subject to prevent reusing
        majority_ages = np.delete(majority_ages, best_match_idx)
        majority_indices = np.delete(majority_indices, best_match_idx)
        
        # Stop if we've used all subjects from majority group (unlikely)
        if (len(majority_ages) == 0) or (len(majority_ages) == len(minority_ages)):
            break
    
    # Get the matched subjects from each group
    majority_matched = majority_df.filter(pl.col("index").is_in(majority_matched_indices))
    
    # Combine the matched groups
    matched_df = pl.concat([minority_df, majority_matched])
    
    # Parameter means after matching
    mean_diff = np.abs(majority_matched[matching_column].mean() - minority_df[matching_column].mean())
    print(f"\nAfter matching -> {minority_name}: {len(minority_matched_indices)}, "
          f"{majority_name} (subsampled): {len(majority_matched_indices)} "
          f"\nMean absolute difference after matching: {mean_diff:.3f}")

    return matched_df.drop("index")


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = _check_args(args)

    # Subsample the majority group
    matched_df = subsample_majority_by_age_match(
        args.dataframe,
        group1=args.group1,
        group2=args.group2,
        matching_column=args.matching_column,
        caliper=args.caliper,
    )

    # Save the matched DataFrame
    out_name = (f"{"matched_" + os.path.basename(args.dataframe_path).split(".")[0]}_{args.group1}_"
                f"{args.group2}_by_{args.matching_column}")
    output_path = os.path.join(args.output, out_name + TERMINATIONS[args.separator])
    
    matched_df.write_csv(output_path, separator=args.separator,
                         include_header=args.has_header)
    print(f"Matched DataFrame saved to {output_path}")


if __name__ == "__main__":
    main()


# Example usage:
# matched_df = subsample_majority_by_age_match(dataframe_path, group_1, group_2, matcher_column, caliper=None)