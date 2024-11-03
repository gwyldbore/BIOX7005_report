import random
import seq_utils

def dummy_blast_results(seq_df, output_df, blast_results):
    # Randomly assign either 'dummy_NR1' or 'dummy_NR4' to each sequence
    # seq_df['blast_results'] = [random.choice(['dummy_NR1', 'dummy_NR4']) for _ in range(len(seq_df))]

    seq_df['blast_prediction'] = seq_df['info'].map(blast_results)

    seq_df.to_csv(output_df, index=False)


def parse_results(filename):

    results = {}

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

    with open(filename) as file:
        for line in file:
            if line.startswith('# Query'): # identifying which sequence

                # get the sequence id from the query line
                seq_id = line.split()[2]

                # set up seq id in results to track count
                results[seq_id] = {'NR1':0, 'NR4':0, 'Other':0}

            # these lines of the blast results aren't useful here
            elif line.startswith('#') or line == '\n' or line.startswith('Fields') or line.startswith('Processing'):
                continue
            
            # otherwise line contains results
            else:
                # line is split by tabs
                fields = line.split('\t')

                # the hit info is index 1 in this search
                match_info = fields[1]
                match_info = match_info.lower()

                # check through the common outputs
                for key in USUAL_BLAST_OUTPUTS.keys():
                    # if output found in result
                    if key in match_info:
                        # increment the appropriate family in results 
                        results[seq_id][USUAL_BLAST_OUTPUTS[key]] += 1
                        # continue
                        break
                    # else:
                    #     # if not one of the found matches, assume its other
                    #     results[seq_id]['Other'] += 1
                    #     # print(f'fields on unmatched: {match_info}')

    # print(f'initial results: {results}')
    # print()


    final_mapping = {}
    for seq_id in results.keys():
        most_frequent = ''
        frequency = -1
        for subfamily in results[seq_id].keys():
            if results[seq_id][subfamily] > frequency:
                most_frequent = subfamily
                frequency = results[seq_id][subfamily]
        final_mapping[seq_id] = most_frequent

    # print(f'final mapping: {final_mapping}')

    return final_mapping






def main():
    fasta = snakemake.input.generated_sequences
    blast_results = snakemake.input.blast_out
    output_df = snakemake.output.blast_df

    seq_df = seq_utils.get_sequence_df(fasta)

    result_map = parse_results(blast_results)

    # these aren't actually dummy results anymore
    # but we love a legacy name
    dummy_blast_results(seq_df, output_df, result_map)


if __name__ == "__main__":
    main()
