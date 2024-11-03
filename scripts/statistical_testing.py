import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import math
from scipy.stats import kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro


def get_initial_category(df):
    return df['overall_prediction'].iloc[0]

def find_first_prediction_changes(df):
    """
    Return the first row where each new `overall_prediction` occurs,
    excluding the initial category and any reversion to it.
    """
    # Identify the initial category from the first row.
    initial_category = get_initial_category(df)
    
    # Shift the `overall_prediction` column to compare with the previous row.
    previous_predictions = df['overall_prediction'].shift(1)

    # Identify where the prediction changes.
    changes = df[df['overall_prediction'] != previous_predictions]

    # Exclude the first row and any rows where the category reverts to the initial one.
    valid_changes = changes[changes['overall_prediction'] != initial_category].iloc[1:]

    # # Keep only the first instance of each unique `overall_prediction` change.
    first_unique_changes = valid_changes.drop_duplicates(subset=['overall_prediction'], keep='first')

    return first_unique_changes


def find_overall_prediction_changes(df):
    """
    Return the first row where each new `overall_prediction` occurs,
    excluding the initial category and any reversion to it.
    """
    # Identify the initial category from the first row.
    initial_category = get_initial_category(df)
    
    # Shift the `overall_prediction` column to compare with the previous row.
    previous_predictions = df['overall_prediction'].shift(1)

    # Identify where the prediction changes.
    changes = df[df['overall_prediction'] != previous_predictions]

    # Exclude the first row and any rows where the category reverts to the initial one.
    valid_changes = changes[changes['overall_prediction'] != initial_category].iloc[1:]

    # # Keep only the first instance of each unique `overall_prediction` change.
    # first_unique_changes = valid_changes.drop_duplicates(subset=['overall_prediction'], keep='first')


        # Create an empty list to store the rows that meet the condition.
    filtered_changes = []

    # Loop over valid changes and check if the next two rows match the current category.
    for idx, row in valid_changes.iterrows():
        current_prediction = row['overall_prediction']
        
        # Check if the next four predictions are the same as the current prediction
        if (idx + 4 < len(df) and 
            df.loc[idx + 1, 'overall_prediction'] == current_prediction and 
            df.loc[idx + 2, 'overall_prediction'] == current_prediction and
            df.loc[idx + 3, 'overall_prediction'] == current_prediction and
            df.loc[idx + 4, 'overall_prediction'] == current_prediction):
            filtered_changes.append(row)
    
    # Convert the list back to a DataFrame
    result_df = pd.DataFrame(filtered_changes)

    # return first_unique_changes
    return result_df



def process_methods_and_replicates(methods):
    """
    Process all methods, each containing multiple replicates, and aggregate results.
    """
    aggregated_results = []

    # Iterate through each method and its replicates
    for method_name, replicates in methods.items():
        for replicate_name, df in replicates.items():
            # Extract first prediction changes for the replicate
            changes = find_first_prediction_changes(df)

            # Add method and replicate information to the dataframe
            changes['method'] = method_name

            # Append the results to the aggregated list
            aggregated_results.append(changes)

    # Combine all results into a single dataframe
    return pd.concat(aggregated_results, ignore_index=True)


def clean_name(name):
    return name.replace('_', ' ')


def get_prediction_order(initial_category):
    base_order = ['NR1', 'NR1-like', 'NR4-like', 'NR4', 'other']

    if initial_category == 'NR1':
        return base_order[1:]
    elif initial_category == 'NR4':
        return ['NR4-like', 'NR1-like', 'NR1', 'other']
    else:
        # default to base
        return base_order
    

#     # Define a function to perform ANOVA or Kruskal-Wallis and post-hoc testing
# def perform_statistical_tests(df, category):
#     """Perform statistical tests for a given category."""
#     # Filter data for the current category
#     category_data = df[df['overall_prediction'] == category]

#     # Check if there are enough unique methods for comparison
#     if category_data['method'].nunique() < 2:
#         print(f"Not enough methods for comparison in category: {category}")
#         return None

#     # Perform Kruskal-Wallis test (non-parametric)
#     kruskal_result = kruskal(
#         *[group['num_mutation'].values for name, group in category_data.groupby('method')]
#     )
#     print(f"Kruskal-Wallis result for {category}: p-value = {kruskal_result.pvalue} \n\n\n")
    

#     if kruskal_result.pvalue < 0.05:
#         # Perform post-hoc Tukey HSD test if significant
#         tukey = pairwise_tukeyhsd(
#             endog=category_data['num_mutation'],
#             groups=category_data['method'],
#             alpha=0.05
#         )
#         print(f"Tukey HSD post-hoc test for {category}:\n{tukey}\n")
#     else:
#         print(f"No significant differences found in {category}.\n")



def run_shapiro_tests(df, outpath):
    """Run Shapiro-Wilk test for each category-method combination and write results to a text file."""
    categories = df['overall_prediction'].dropna().unique()
    methods = df['method'].dropna().unique()

    results = []

    # Iterate over categories and methods
    for category in categories:
        for method in methods:
            # Filter data for the current category and method
            method_data = df[(df['overall_prediction'] == category) & 
                             (df['method'] == method)]['num_mutation']
            
            # print(f'shapiro method data for {method}, {category}: {method_data}')

            # if not method_data.empty:
            if len(method_data) >=3:
                # Run the Shapiro-Wilk test
                stat, p_value = shapiro(method_data)
                results.append({
                    'Category': category,
                    'Method': method,
                    'W-Statistic': stat,
                    'p-value': p_value
                })
            else:
                results.append({
                    'Category': category,
                    'Method': method,
                    'W-Statistic': "insufficient data",
                    'p-value': 'insufficient data'
                })

    # Write the results to a text file with proper formatting
    with open(outpath, 'w') as f:
        f.write("Shapiro-Wilk Test Results\n")
        f.write("=" * 40 + "\n")
        for result in results:
            f.write(f"Category: {result['Category']}, Method: {result['Method']}\n")
            normal = 'False'
            # check that pvalue is actually calculated
            if (not isinstance(result['p-value'], str)):
                if result['p-value'] > 0.05:
                    normal = 'True'
                f.write(f"W-Statistic: {result['W-Statistic']:.4f}, p-value: {result['p-value']:.4e}, normal: {normal}\n")
            # otherwise if there was insufficient data to calculate
            else:
                f.write(f"W-Statistic: {result['W-Statistic']}, p-value: {result['p-value']}, normal: {normal}\n")
            f.write("-" * 40 + "\n")

def run_kruskal_wallis(df, outpath):
    """Run Kruskal-Wallis test for each category and write results to a text file."""
    categories = df['overall_prediction'].unique().dropna()
    results = []

     # Set 'method' as categorical with all unique levels in the main dataframe
    df['method'] = pd.Categorical(df['method'], categories=df['method'].unique())



    # Write the results to a text file
    with open(outpath, 'w') as f:
        f.write("Kruskal-Wallis Test Results\n")
        f.write("=" * 40 + "\n")

            
        # Iterate over categories
        for category in categories:
            # Group by method within the category
            print(f'processing {category} right now')
            category_data = df[df['overall_prediction'] == category]

            category_data['method'] = pd.Categorical(category_data['method'], categories=df['method'].cat.categories)


            # # Prepare data for Kruskal-Wallis test (grouped by method)
            # grouped_data = [
            #     group['num_mutation'].values for _, group in category_data.groupby('method')
            # ]

            grouped_data = [
                group['num_mutation'].values for _, group in category_data.groupby('method') if len(group) > 0
            ]

            if len(grouped_data) < 2:
                print(f"Not enough methods for comparison in category: {category}")
                continue

            # Run the Kruskal-Wallis test
            stat, p_value = kruskal(*grouped_data)

            print(f"Kruskal-Wallis result for {category}: p-value = {p_value} \n\n")
        

            # # Store the result
            # results.append({
            #     'Category': category,
            #     'Statistic': stat,
            #     'p-value': p_value,
            # })

        

            # for result in results:
            f.write(f"Category: {category}\n")
            significant = False
            if p_value <= 0.05:
                significant = True
            f.write(f"Statistic: {stat:.4f}, p-value: {p_value:.4e}, significant: {significant}\n")

            # if a category is significant, do post-hoc dunns test 
            if significant:
                f.write("Dunn's Post-hoc Test Results:\n")

                print(f"Dunn's test for category '{category}' with methods: {category_data['method'].unique()}\n\n\n")


                # Run Dunn's test with Bonferroni correction
                dunn_results = sp.posthoc_dunn(
                    category_data, val_col='num_mutation', group_col='method', p_adjust='bonferroni'
                )

                # Write Dunn's test results to file in table format
                f.write(dunn_results.to_string())
                f.write("\n\n")


            f.write("-" * 40 + "\n")



def plot_qq_grid(df, outpath):
    """Generate a grid of Q-Q plots with categories as rows and methods as columns."""
    # Get unique categories and methods
    categories = df['overall_prediction'].dropna().unique()
    methods = df['method'].dropna().unique()
    # print(f'methods: {methods}')
    # print(f'categories: {categories}')

    # Dynamically adjust the grid size based on valid combinations
    num_categories = len(categories)
    num_methods = len(methods)

    if num_categories == 0 or num_methods == 0:
        print("No valid data to plot.")
        return

    # Create the grid: categories in rows, methods in columns
    fig, axes = plt.subplots(
        num_categories, num_methods, figsize=(6 * num_methods, 6 * num_categories),
        squeeze=False  # Ensure we always get a 2D array of axes
    )
    fig.suptitle('Q-Q Plots for All Categories and Methods', fontsize=18, fontweight='bold')

    # Iterate over categories and methods to populate the grid
    for i, category in enumerate(categories):
        for j, method in enumerate(methods):
            ax = axes[i, j]  # Select the correct axis

            # Filter data for the current category and method
            method_data = df[(df['overall_prediction'] == category) & (df['method'] == method)]['num_mutation']

            if method_data.empty:
                # If no data, disable the axis and display a message
                ax.axis('off')
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            else:
                # Generate the Q-Q plot
                stats.probplot(method_data, dist="norm", plot=ax)
                ax.set_title(f'{category} - {method}', fontsize=12)

    # Adjust layout to prevent overlap and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outpath)
    plt.close()


def main():
    input_files = snakemake.input


    grouped_first_results = []
    grouped_results = []
    all_data = []

    for file in input_files:
        df = pd.read_csv(file)
        first_changes = find_first_prediction_changes(df)
        actual_changes = find_overall_prediction_changes(df)

        grouped_first_results.append(first_changes)
        grouped_results.append(actual_changes)
        all_data.append(df)

    # can just run this on the last one as it'll be the same for all of them
    initial_category = get_initial_category(df)
    order = get_prediction_order(initial_category)
    
    # create the overall results group
    grouped_df = pd.concat(grouped_results, ignore_index=True)
    # grouped_df.to_csv('TESTFILE.csv')
    grouped_df['method'] = grouped_df['method'].apply(clean_name)
    grouped_df['overall_prediction'] = pd.Categorical(
        grouped_df['overall_prediction'], categories=order, ordered=True
    )

    grouped_df.to_csv(snakemake.output.multi_df)

    # create the first change only group
    grouped_first_df = pd.concat(grouped_first_results, ignore_index=True)
    grouped_first_df['method'] = grouped_first_df['method'].apply(clean_name)
    grouped_first_df['overall_prediction'] = pd.Categorical(
        grouped_first_df['overall_prediction'], categories=order, ordered=True
    )

    grouped_first_df.to_csv(snakemake.output.first_df)




    # CREATE FIRST CHANGE BOXPLOT
    # Create a box plot for each overall_prediction category
    g = sns.catplot(
        data=grouped_first_df,
        x='method',
        y='num_mutation',
        col='overall_prediction',  # Create separate plots for each category
        kind='box',
        height=7,  # Adjust the height of each plot
        aspect=0.8  # Control the aspect ratio of each plot
    )

    # Remove individual x-axis labels from each subplot
    for ax in g.axes.flat:
        # Extract the original category name from the title
        original_title = ax.get_title().split(' = ')[1]  # Extract the prediction value
        # Set the cleaned title (with underscores replaced by spaces)
        ax.set_title(clean_name(original_title), fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('')

        # Rotate x-axis labels for readability
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

    # Adjust the title and labels
    g.set_axis_labels('Method', 'Number of Mutations')
    g.figure.subplots_adjust(bottom=0.25, top=0.88)
    g.set(ylim=(0, 150))
    g.figure.suptitle("Mutation Counts at First Family Prediction Change by Method", fontweight='bold', fontsize='x-large')
    plt.savefig(snakemake.output.boxplot_first)
    plt.close() # close to save memory



    # Create a box plot for each overall_prediction category
    g = sns.catplot(
        data=grouped_df,
        x='method',
        y='num_mutation',
        col='overall_prediction',  # Create separate plots for each category
        kind='box',
        height=7,  # Adjust the height of each plot
        aspect=0.8  # Control the aspect ratio of each plot
    )
    

    # Remove individual x-axis labels from each subplot
    for ax in g.axes.flat:
        # Extract the original category name from the title
        original_title = ax.get_title().split(' = ')[1]  # Extract the prediction value
        # Set the cleaned title (with underscores replaced by spaces)
        ax.set_title(clean_name(original_title), fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('')

        # Rotate x-axis labels for readability
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

    # Adjust the title and labels
    g.set_axis_labels('Method', 'Number of Mutations')
    g.figure.suptitle("Mutation Counts at Overall Family Prediction Change by Method", fontsize='x-large', fontweight='bold')
    g.set(ylim=(0, 150))
    g.figure.subplots_adjust(bottom=0.25, top=0.88)
    plt.savefig(snakemake.output.boxplot_multi)
    plt.close() # close to save memory





    # Get the unique categories from the data
    categories = grouped_df['overall_prediction'].unique().dropna()

    plot_qq_grid(grouped_df, snakemake.output.qqplot)

    # Run Shapiro-Wilk tests and write the results to a text file
    run_shapiro_tests(grouped_df, snakemake.output.shapiro)

    run_kruskal_wallis(grouped_df, snakemake.output.kruskal)



    # get the stats for the first change method too:
    plot_qq_grid(grouped_first_df, snakemake.output.qqplot_first)

    # Run Shapiro-Wilk tests and write the results to a text file
    run_shapiro_tests(grouped_first_df, snakemake.output.shapiro_first)

    run_kruskal_wallis(grouped_first_df, snakemake.output.kruskal_first)





    # CREATE A VERSION THAT PLOTS ALL OF THE VALUES FOR A FAMILY NOT JUST WHERE IT CHANGES

    combined_df = pd.concat(all_data, ignore_index=True)

    # Clean method names if not already cleaned
    combined_df['method'] = combined_df['method'].apply(clean_name)

    # Define the categorical order for `overall_prediction`
    initial_category = get_initial_category(combined_df)
    if initial_category == "NR1":
        order = ["NR1", "NR1-like", "NR4-like", "NR4", "other"]
    else:
        order = ['NR4', 'NR4-like', 'NR1-like', 'NR1', 'other']

    # Set order for plotting
    combined_df['overall_prediction'] = pd.Categorical(combined_df['overall_prediction'], categories=order, ordered=True)

    # Plot range of values across predictions grouped by method
    plt.figure(figsize=(10, 8))
    g = sns.catplot(
        data=combined_df,
        x='method',
        y='num_mutation',
        col='overall_prediction',  # Separate plot for each prediction
        kind='box',
        height=6,
        aspect=0.7
    )

    # Adjust layout and labels
    for ax in g.axes.flat:
        ax.set_xlabel('')
        ax.set_title(ax.get_title().replace('overall_prediction = ', ''))

        # Rotate x-axis labels for readability
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

    # Set main plot title
    g.set_axis_labels('Method', 'Number of Mutations')
    g.figure.suptitle("Mutation Counts Across All Predictions by Method", fontsize='x-large', fontweight='bold')
    g.figure.subplots_adjust(top=0.88, bottom=0.25)  # Adjust to fit title

    # Save the plot
    plt.savefig(snakemake.output.boxplot_combined)
    plt.close()

    # Run additional statistical tests if desired
    # plot_qq_grid(grouped_all_values, snakemake.output.qqplot_combined)
    # run_shapiro_tests(grouped_all_values, snakemake.output.shapiro_combined)
    # run_kruskal_wallis(grouped_all_values, snakemake.output.kruskal_combined)

   











    # # method_name = snakemake.wildcards.method_name

    # replicates = [pd.read_csv(file) for file in input_files]
    # combined_df = pd.concat(replicates, ignore_index=True)


    # # make a boxplot of the stats
    # # plt.figure(figsize=(10, 6))
    # # sns.boxplot(x='method', y='num_mutation', data=combined_df)
    # # plt.title(f'Mutation Counts by {method_name}')
    # # plt.ylabel("Number of mutations")
    # # plt.savefig(snakemake.output.boxplot)
    # # plt.close() # close to save memory

    # # Get unique methods for plotting
    # methods = combined_df['method'].unique()

    # # Calculate the number of rows and columns for the grid layout
    # num_methods = len(methods)
    # cols = 3  # Define the number of columns in the grid
    # rows = math.ceil(num_methods / cols)  # Calculate the number of rows needed

    # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)

    # # Generate boxplots for each method
    # for i, method in enumerate(methods):
    #     row, col = divmod(i, cols)  # Calculate grid position
    #     sns.boxplot(
    #         x='method', y='num_mutation', 
    #         data=combined_df[combined_df['method'] == method], ax=axes[row, col]
    #     )
    #     axes[row, col].set_title(f'Mutation Counts for {method}')

    # plt.ylabel("Number of mutations")
    # plt.savefig(snakemake.output.boxplot)
    # plt.close() # close to save memory



    # # 3.1 Shapiro-Wilk test for normality
    # for method, data in combined_df.groupby('method'):
    #     stat, p = stats.shapiro(data['num_mutation'])
    #     print(f'{method} - Shapiro-Wilk test p-value: {p}')

    # # 3.2 Levene’s test for equal variances
    # stat, p = stats.levene(
    #     *[group['num_mutation'].values for name, group in combined_df.groupby('method')]
    # )
    # print(f'Levene’s test p-value: {p}')

    # # 3.3 Hypothesis Testing (ANOVA or Kruskal-Wallis)
    # if p > 0.05:  # If variances are equal
    #     stat, p = stats.f_oneway(
    #         *[group['num_mutation'].values for name, group in combined_df.groupby('method')]
    #     )
    #     print(f'ANOVA test p-value: {p}')
    #     if p < 0.05:
    #         tukey = pairwise_tukeyhsd(combined_df['num_mutation'], combined_df['method'])
    #         print(tukey)
    # else:
    #     stat, p = stats.kruskal(
    #         *[group['num_mutation'].values for name, group in combined_df.groupby('method')]
    #     )
    #     print(f'Kruskal-Wallis test p-value: {p}')
    #     if p < 0.05:
    #         dunn = sp.posthoc_dunn(combined_df, val_col='num_mutation', group_col='method', p_adjust='bonferroni')
    #         print(dunn)

    # # Step 4: Chi-square Test for Mutated Positions
    # positions_df = pd.crosstab(combined_df['mutated_positions'], combined_df['method'])
    # chi2, p, _, _ = stats.chi2_contingency(positions_df)
    # print(f'Chi-square test p-value: {p}')









if __name__ == "__main__":
    main()