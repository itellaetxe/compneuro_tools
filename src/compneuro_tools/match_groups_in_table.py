import os

from argparse import ArgumentParser

import numpy as np
import polars as pl

from scipy.stats import ttest_ind


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
        "--no_size_match",
        action="store_true",
        required=False,
        help="If True, match the sizes of the two groups. Default is False.",
    )
    parser.add_argument(
        "--pvalue_threshold",
        type=float,
        required=False,
        default=0.05,
        help="P-value threshold for statistical significance. Default is 0.05.",
    )

    parser.add_argument(
        "--categorical",
        action="store_true",
        required=False,
        help="Indicates if the matching column is categorical. Default is False (numerical).",
    )

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


def subsample_majority_by_categorical_match(df: pl.DataFrame, #TODO: implement this function
                                    group1: str,
                                    matching_column: str,
                                    match_sizes: bool = False,
                                    pvalue_threshold: float = 0.05,) -> pl.DataFrame:
    """
    In a given group, subsample the majority category until it matches the minority category
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing participant data
    group1 : str
        Column name for the first group (e.g., "g1_SN")
    group2 : str
        Column name for the second group (e.g., "g2_PV")
    matching_column : str
        Column name to use for matching
    match_sizes : bool, optional
        If True, match the sizes of the two groups. Default is False.
    pvalue_threshold : float, optional
        P-value threshold for statistical significance (default is 0.05)
        
    Returns:
    --------
    matched_df : polars.DataFrame
        DataFrame containing participants from both groups after matching
    """
    pass
    # # Extract participants from each group
    # group1_df = df.filter(pl.col(group1) == "1")
    
    # # Determine which is minority and majority group
    # # Find the category with the maximum count in the majority group
    # max_category = majority_counts[majority_counts == majority_counts.max()]

    # # Remove rows from the majority group until the sizes are equal
    # while pval >= pvalue_threshold:
    #     # Get the counts of each category in the matching column
    #     minority_counts = minority_df[matching_column].value_counts()
    #     majority_counts = majority_df[matching_column].value_counts()

        

    #     # Remove rows from the majority group that match the maximum category
    #     majority_df = majority_df.filter(pl.col(matching_column) != max_category.index[0])

    #     # Check if the sizes are equal
    #     if len(minority_df) == len(majority_df):
    #         break

    # mean_diff = np.abs(minority_df[matching_column].mean() - majority_df[matching_column].mean())
    # print(f"Original sizes -> {minority_name} (minority): {len(minority_df)}, {majority_name} (majority): {len(majority_df)}"
    #       f"\nMean absolute difference in \"{matching_column}\" before matching: {mean_diff:.3f}\n")



    # pval = chisquare(minority_df[matching_column].value_counts().to_numpy(),
    #                  majority_df[matching_column].value_counts().to_numpy()).pvalue
    # print(f"\nAfter matching -> {minority_name} (minority): {len(minority_df)}, "
    #       f"{majority_name} (subsampled majority): {len(majority_matched_rows)} "
    #       f"\nMean absolute difference in \"{matching_column}\" after matching: {mean_diff:.3f}"
    #       f"\np-value: {pval:.3f}\n")

    # return matched_df.drop("index")


def subsample_majority_by_numerical_match(df: pl.DataFrame,
                                    group1: str,
                                    group2: str,
                                    matching_column: str,
                                    match_sizes: bool = False,
                                    pvalue_threshold: float = 0.05,
                                    caliper: float = None) -> pl.DataFrame:
    """
    Subsample participants from the majority group by finding specific NUMERICAL column
    matches in the minority group.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing participant data
    group1 : str
        Column name for the first group (e.g., "g1_SN")
    group2 : str
        Column name for the second group (e.g., "g2_PV")
    matching_column : str
        Column name to use for matching
    match_sizes : bool, optional
        If True, match the sizes of the two groups. Default is False.
    pvalue_threshold : float, optional
        P-value threshold for statistical significance (default is 0.05)
    caliper : float, optional
        Maximum age difference allowed for matching (in years)
        
    Returns:
    --------
    matched_df : polars.DataFrame
        DataFrame containing participants from both groups after matching
    """

    # Extract participants from each group
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
    majority_matched_rows = []
    
    # Get ages as numpy arrays for faster processing
    minority_ages = minority_df[matching_column].to_numpy()

    mean_diff = np.abs(minority_df[matching_column].mean() - majority_df[matching_column].mean())
    print(f"Original sizes -> {minority_name} (minority): {len(minority_df)}, {majority_name} (majority): {len(majority_df)}"
          f"\nMean absolute difference in \"{matching_column}\" before matching: {mean_diff:.3f}\n")
    
    # For each participant in minority group, find closest match in majority group
    broken = False
    for minority_age in minority_ages:
        majority_df = majority_df.with_columns(
            (majority_df[matching_column] - minority_age).abs().alias("age_diff")
        )
        age_diffs = majority_df["age_diff"]
        
        # If using a caliper, skip if no matches within caliper
        if (caliper is not None) and (age_diffs.min() > caliper):
            print(f"### No matches within caliper ({caliper}) for age {minority_age:.3f}"
                  f"({age_diffs.min():.3f} > {caliper}). Cannot continue matching.")
            broken = True
            break

        elif age_diffs.min() > minority_df[matching_column].std():
            print(f"### Adding participant with age difference higher than minority "
                  f"group age standard deviation ({age_diffs.min():.3f} > {minority_df[matching_column].std():.3f})")

        # Find the index of the best match
        best_match_idx = majority_df["age_diff"].arg_min()

        majority_matched_rows.append(majority_df[best_match_idx])
        majority_matched_age = pl.concat(majority_matched_rows, how="vertical")[matching_column]
        
        # Remove the matched participant to prevent reusing
        majority_df = majority_df.drop("index").with_row_index().filter(pl.col("index") != best_match_idx)
        
        # Stop when all participants in the majority group are matched, or when there is a statistically significant difference in the ages
        t_test = ttest_ind(minority_ages, majority_matched_age, equal_var=False)
        pval = t_test.pvalue
        if np.isnan(pval):
            pval = 1.0

        # First we keep adding participants until reaching the minority group size
        # Then we check if the p-value is significant. Keep adding participants until the p-value is significant
        if (pval < pvalue_threshold) and (len(majority_matched_rows) == len(minority_ages)):
            print(f"Stopping matching due to significant difference in ages (p = {t_test.pvalue:.3f})")
            majority_matched_rows = majority_matched_rows[:-1]
            broken = True
            break

    # If match_sizes is False, keep adding participants until the p-value indicates a statistically significant difference
    if (not match_sizes) and (not broken):
        # Compute the mean of the minority group
        minority_mean = minority_df[matching_column].mean()
        # Compute the age differences
        majority_df = majority_df.with_columns(
            (majority_df[matching_column] - minority_mean).abs().alias("age_diff")
        )
        age_diffs = majority_df["age_diff"]

        # While p-value not significant and still participants in majority group or caliper not reached, keep adding participants
        while ((pval >= pvalue_threshold) and len(majority_df) > 0) or ((age_diffs).min() < caliper):
            # Find the index of the best match and add it to the matched indices
            best_match_idx = majority_df["age_diff"].arg_min()
            majority_matched_rows.append(majority_df[best_match_idx])
            majority_matched_age = pl.concat(majority_matched_rows, how="vertical")[matching_column]

            # Compute the new p-value
            t_test = ttest_ind(minority_ages, majority_matched_age, equal_var=False)
            pval = t_test.pvalue

            if pval < pvalue_threshold:
                print(f"Stopping matching due to significant difference in ages (p = {pval:.3f})")
                # Remove the last index because it introduces a significant difference
                majority_matched_rows = majority_matched_rows[:-1]
                break

            # Remove the matched participant to prevent reusing
            majority_df = majority_df.drop("index").with_row_index().filter(pl.col("index") != best_match_idx)
            majority_df = majority_df.with_columns(
            (majority_df[matching_column] - minority_mean).abs().alias("age_diff")
            )
            age_diffs = majority_df["age_diff"]


    # Get the matched participants from each group
    majority_matched = pl.concat(majority_matched_rows, how="vertical").drop("age_diff")
    majority_matched_age = majority_matched[matching_column]
    
    # Combine the matched groups
    matched_df = pl.concat([minority_df, majority_matched])
    
    # Parameter means after matching
    pval = ttest_ind(minority_ages, majority_matched_age, equal_var=False).pvalue
    mean_diff = np.abs(majority_matched[matching_column].mean() - minority_df[matching_column].mean())
    print(f"\nAfter matching -> {minority_name} (minority): {len(minority_df)}, "
          f"{majority_name} (subsampled majority): {len(majority_matched_rows)} "
          f"\nMean absolute difference in \"{matching_column}\" after matching: {mean_diff:.3f}"
          f"\np-value: {pval:.3f}\n")

    return matched_df.drop("index")


def main():
    print("----------------------------------------------------------")
    parser = _setup_parser()
    args = parser.parse_args()
    args = _check_args(args)

    # Subsample the majority group
    if args.categorical:
        matched_df = subsample_majority_by_categorical_match(
            args.dataframe,
            group1=args.group1,
            group2=args.group2,
            matching_column=args.matching_column,
            match_sizes=(not args.no_size_match),
            pvalue_threshold=args.pvalue_threshold,
        )
    else:
        matched_df = subsample_majority_by_numerical_match(
            args.dataframe,
            group1=args.group1,
            group2=args.group2,
            matching_column=args.matching_column,
            match_sizes=(not args.no_size_match),
            pvalue_threshold=args.pvalue_threshold,
            caliper=args.caliper,
        )

    # Save the matched DataFrame
    out_name = (f"matched_{os.path.basename(args.dataframe_path).split('.')[0]}_{args.group1}_"
                f"{args.group2}_by_{args.matching_column}")
    output_path = os.path.join(args.output, out_name + TERMINATIONS[args.separator])
    
    matched_df.write_csv(output_path, separator=args.separator,
                         include_header=args.has_header)
    print(f"Matched DataFrame saved to {output_path}")
    print("----------------------------------------------------------")
    


if __name__ == "__main__":
    main()


# Example usage:
# matched_df = subsample_majority_by_age_match(dataframe_path, group_1, group_2, matcher_column, caliper=None)