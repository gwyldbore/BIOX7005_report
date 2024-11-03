import pandas as pd
import pickle

def load_interproscan_df(path):

    column_names = [
        "info",
        "label",
        "sub_label",
        "description",
        "start",
        "stop",
        "score",
        "status",
        "date",
        "extended_description"
    ]
    selected_columns = [0, 3, 4, 5, 6, 7, 8, 9, 10, 12]

    interpro_df = pd.read_csv(path, sep='\t', usecols=selected_columns, names=column_names)
    return interpro_df


def parse_prints_output(row):
    USUAL_BLAST_OUTPUTS = {'subfamily 4': 'NR4',
                           'sub 4': 'NR4', '4 ': 'NR4',
                           'hr38': 'NR4', ' 38 ': 'NR4',
                           'hzf-3': 'NR4', 'nurr1': 'NR4',
                           'nur77': 'NR4', 'nr4': 'NR4',
                           'nor1': 'NR4', 

                           'subfamily 1': 'NR1', 
                           'retinoic acid receptor': 'NR1',
                           'oxysterol': 'NR1', 'LXR': 'NR1',

                           'probable nuclear hormone receptor': 'Other',
                           'nuclear receptor related 1': 'Other',
                           'nuclear receptor isoform x1': 'Other',
                           'nuclear hormone receptor e75': 'Other',
                           'nuclear receptor of the nerve growth': 'Other',
                           'ligand-binding domain': 'Other',
                           'subfamily 2': 'Other'
                           }
    
    # set up results dict to track count
    results = {'NR1':0, 'NR4':0, 'Other':0}
    
    # get the prints data
    try:
        all_results = row['PRINTS']
    except KeyError:
        all_results = row['Gene3D']

    # # which column contains the sequence id?
    # seq_id = row['']

    # split each result, separated by ;
    split_results = all_results.split('; ')
    # print(f'results: {split_results}')

    for result in split_results:
        result = result.lower()

        for key in USUAL_BLAST_OUTPUTS.keys():
            if key in result:
                # increment the appropriate family in results 
                results[USUAL_BLAST_OUTPUTS[key]] += 1
                break

    most_frequent = ''
    frequency = -1
    for subfamily in results.keys():
        
        # print(f'subfamily {subfamily}, most freq {most_frequent}:{frequency}')

        if results[subfamily] > frequency:
            most_frequent = subfamily
            frequency = results[subfamily]
    
    return most_frequent




def load_mutation_positions(filepath):

    with open(filepath, 'r') as file:
        all_mutations = []
        for line in file:
            line = line.strip()
            if line == '':
                all_mutations.append([])
            else:
                line_array = line.split(',')
                line_array = [int(pos) for pos in line_array]
                all_mutations.append(line_array)
            # all_mutations.append(line.split(','))

    return all_mutations


def get_mutation_positions(mutation_list, index):
    return mutation_list[int(index)]




def main():
    # Load the dataframes
    blast_df = pd.read_csv(snakemake.input.blast_df)
    interproscan_df = load_interproscan_df(snakemake.input.interproscan_df)

    with open(snakemake.input.embedding_df, "rb") as input_file:
        embedding_df = pickle.load(input_file)

    merged_df = embedding_df.merge(interproscan_df, on='info', how='left')

    pivot_df = merged_df.pivot_table(index='info', columns='label', values='extended_description',
                                     aggfunc=lambda x: '; '.join(x))
    pivot_df = pivot_df.reset_index()

    # Load logistic regression predictions
    logistic_predictions_df = pd.read_csv(snakemake.input.logreg_results)

    # Merge back with embedding dataframe
    final_df = embedding_df.merge(pivot_df, on=['info'], how='left')

    # Merge with blast results
    final_df = final_df.merge(blast_df[['info', 'blast_prediction']], on='info', how='left')

    # Merge with logistic regression predictions
    # print(logistic_predictions_df)
    final_df = final_df.merge(logistic_predictions_df[['info', 'embedding_prediction']], on='info', how='left')

    # final_df['has_subfamily_4'] = final_df['PRINTS'].str.contains('subfamily 4', na=False)
    # final_df['has_subfamily_1'] = final_df['PRINTS'].str.contains('subfamily 1', na=False)
    # if 'PRINTS' in final_df.columns():
    final_df['interproscan_prediction'] = final_df.apply(parse_prints_output, axis=1)

    # also add the mutation position list
    mutation_positions = load_mutation_positions(snakemake.input.mutationfile)
    # because why not while we're updating the dataframe in one place
    final_df['mutated_positions'] = final_df['num_mutation'].apply(
        lambda x: get_mutation_positions(mutation_positions, x)
    )

    # Save the merged dataframe
    final_df.to_csv(snakemake.output.merged_df, index=False)


if __name__ == "__main__":
    main()
