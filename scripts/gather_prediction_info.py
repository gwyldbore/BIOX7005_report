import pandas as pd

def overall_prediction(row):
    NR1count = 0
    NR4count = 0
    othercount = 0

    for prediction in ['blast_prediction', 'interproscan_prediction', 'embedding_prediction']:
        if row[prediction] == 'NR1':
            NR1count += 1
        elif row[prediction] == 'NR4':
            NR4count += 1
        else:
            othercount += 1

    if NR1count == 3:
        return 'NR1'
    elif NR1count == 2:
        return 'NR1-like'
    elif NR4count == 3:
        return 'NR4'
    elif NR4count == 2:
        return 'NR4-like'
    else:
        return 'other'
    
def insert_method_type(row):
    return snakemake.wildcards.method_name

def main():
    # load in the merged df
    merged_df = pd.read_csv(snakemake.input.merged_df)

    results_df = merged_df[['info', 'sequence', 'num_mutation', 
                            'blast_prediction', 'interproscan_prediction', 
                            'embedding_prediction', 'mutated_positions']].copy()
    
    results_df['overall_prediction'] = results_df.apply(overall_prediction, axis=1)
    results_df['method'] = results_df.apply(insert_method_type, axis=1)

    results_df.to_csv(snakemake.output.results_df, index=False)


if __name__ == "__main__":
    main()

