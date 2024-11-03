from Bio import SeqIO
from Bio.Seq import Seq, MutableSeq
from Bio.SeqRecord import SeqRecord
import random
import pandas as pd
import numpy as np


def calculate_differences(seq1, seq2):
    """
    Calculate the different positions between the two provided sequences, and 
    identify characters to mutate to at those positions.

    Params: 
        seq1: SeqRecord object containing the origin sequence
        seq2: SeqRecord object containing the target sequence

    Returns:
        List[tuple(int, char)] lsit of tuples (position index, target character)
    """

    pos_mutation = []

    for i, character in enumerate(seq1.seq):
        # print(character, seq2.seq[i])
        if character != seq2.seq[i]:

            pos_mutation.append((i, seq2.seq[i]))

    return pos_mutation


def get_specified_mutations(seq1, seq2, positions):
    """
    Retrieve the characters to mutate to at specific positions, if they differ 
    from the origin sequence.

    Params: 
        seq1: SeqRecord object containing the origin sequence
        seq2: SeqRecord object containing the target sequence

    Returns:
        List[tuple(int, char)] lsit of tuples (position index, target character)
    """
    pos_mutation = []

    for position in positions:
        if seq1.seq[position] != seq2.seq[position]:
            pos_mutation.append((position, seq2.seq[position]))

    return pos_mutation


def remove_gaps(sequence):
    """
    Removes gaps from a sequence and returns the ungapped sequence
    """

    ungapped = ''
    for character in sequence.seq:
        if character == '-':
            continue
        else:
            ungapped += character

    record = SeqRecord(
        Seq(ungapped),
        id=f"{sequence.id}",
        description=''
    )

    return record

def remove_common_gaps(seq1, seq2):
    """Removes gaps which are common to both sequences, preserving alignment."""

    seq1_out, seq2_out = '', ''
    removed_at = []


    for i, character in enumerate(seq1.seq):
        if character == '-' and seq2.seq[i] == '-':
            removed_at.append(i)
            continue
        else:
            seq1_out += character
            seq2_out += seq2.seq[i]

    seqrec1 = SeqRecord(
        Seq(seq1_out),
        id=f"{seq1.id}",
        description='')
    
    seqrec2 = SeqRecord(
        Seq(seq2_out),
        id=f"{seq2.id}",
        description='')
    
    return seqrec1, seqrec2, removed_at



def get_nonconservative_mutations(seq1, seq2):
    """
    Get mutations along the whole sequence only if the mutation causes
    the amino acide to change types, e.g. basic to acidic etc. 
    """
    AMINO_TYPES = {'A':'aliphatic', 'G':'aliphatic', 'I':'aliphatic', 
                   'L':'aliphatic', 'P':'aliphatic', 'V':'aliphatic', 
                   'F':'aromatic', 'W':'aromatic', 'Y':'aromatic', 
                   'R':'basic', 'H':'basic', 'K':'basic', 
                   'D':'acidic', 'E':'acidic', 'S':'', 
                   'T':'polar', 'C':'polar', 'M':'polar', 
                   'N':'polar', 'Q':'polar', '-':'gap'}

    pos_mutation = []

    for i, seq1_char in enumerate(seq1.seq):
        seq2_char = seq2.seq[i]

        char1_type = AMINO_TYPES[seq1_char]
        char2_type = AMINO_TYPES[seq2_char]

        if char1_type != char2_type:
            pos_mutation.append((i, seq2_char))

    return pos_mutation



def assign_priority(row): 
    """Assign the mutation priorities to the row of the dataframe based on compared conservation."""  

    # Extract values for comparison
    vc1 = set(row['very_conserved_positions_origin'])
    vc4 = set(row['very_conserved_positions_target'])
    c1 = set(row['conserved_positions_origin'])
    c4 = set(row['conserved_positions_target'])

    if (vc1 and vc4): # if both have a highly conserved site
        # the two are the same - case 1
        if vc1 == vc4:
            return 10
            # don't care about anything else, this is lowest priority

        # the two are different 
        else:
            # one overlaps with the other's conserved - mid priority - case 2
            if (vc1.intersection(c4) or vc4.intersection(c1)):
                return 2

            # no overlap - highest priority - case 3
            else:
                return 1

    elif (not vc1 and not vc4): # neither have highly conserved site

        # both have conserved
        if (c1 and c4):
            # conserved are the same - low priority - case 4
            if c1 == c4:
                return 8

            # conserved intersect - mid priority - case 5
            elif c1.intersection(c4):
                return 6

            # conserved are unique - higher priority - case 6
            else:
                return 4

        # only one has conserved - case 7
        elif c1 or c4:
            return 7
            # mid-low priority

        # neither have conserved - case 8 (neither have anything)
        else:
            return 10
            # this position literally does not matter because there's no conservation


    else: # one or the other has a highly conserved site (vc1 or vc4)

        # both have conserved
        if c1 and c4:

            # conserved is the same - case 9
            if c1 == c4:
                return 6 # means same position exists in c and vc but one has it higher
                # so this is reasonably low priority

            # conserved intersects
            
                # with very conserved - case 10
            elif c1.intersection(vc4) or c4.intersection(vc1):
                # if conserved is same as other very conserved - conservation has mostly remained so low priority
                if (c1 == vc4) or (c4 == vc1):
                    return 6
                # else conserved just intesects with very conserved
                else:
                    return 5

                # with conserved - case 11
            elif c1.intersection(c4):
                return 4

            # conserved is unique - case 12
            else:
                return 3


        # neither have conserved
            # literally impossible 

        # only one has conserved - case 13
            # has to be same one as highly conserved so not helpful/important
            # this is gonna be more important going one direction vs the other
            # leaving it higher for now because going from vc to nothing is a big change
        else:
            return 3
        
def parse_filename(inputfile):
        """helper to get the dataset and sequences out of filename"""
        removedpath = inputfile.split('/')[-1]
        removedextension = removedpath.split('.')[0]
        name_parts = removedextension.split('_')

        dataset = name_parts[0]
        originseq = name_parts[2]
        targetseq = name_parts[3]
        return dataset, originseq, targetseq



def get_probabilistic_mutations(inputfile, removed_at):
    dataset, originseq, targetseq = parse_filename(inputfile)

    # because Index is 1 indexed and removed at was 0 indexed, add 1 to all of them
    removed_at = [x+1 for x in removed_at]

    datapath = '/'.join(inputfile.split('/')[:-1])
    originseq_file = f'{datapath}/{dataset}_{originseq}_marginal.tsv'
    targetseq_file = f'{datapath}/{dataset}_{targetseq}_marginal.tsv'

    df_origin = pd.read_csv(originseq_file, sep='\t')
    df_target = pd.read_csv(targetseq_file, sep='\t')

    """
    nan_rows_origin = df_origin[df_origin.drop(columns=['Index']).isna().all(axis=1)]
    nan_rows_target = df_target[df_target.drop(columns=['Index']).isna().all(axis=1)]

    common_indices_origin = nan_rows_origin['Index'].isin(nan_rows_target['Index'])
    common_indices_target = nan_rows_target['Index'].isin(nan_rows_origin['Index'])

    # this removes the common gaps so that everything lines up with my other sequences
    # Filter out the rows from origin that match the common 'Index' values
    df_origin_cleaned = df_origin[~(df_origin['Index'].isin(nan_rows_origin[common_indices_origin]['Index']))]
    # filter target similarly
    df_target_cleaned = df_target[~(df_target['Index'].isin(nan_rows_target[common_indices_target]['Index']))]

    """

    # a workaround for the marginal distribution having different shit to the actual seq
    df_origin_cleaned = df_origin[~df_origin['Index'].isin(removed_at)]
    df_target_cleaned = df_target[~df_target['Index'].isin(removed_at)]

    # now fix index in both dfs to be 0 indexed and sequential based on sequence
    i = 0

    # # Create a single string by concatenating the highest-value column names across all rows
    # result_string = ''.join(
    #     df_origin.drop(columns=['Index']).apply(
    #     lambda row: '-' if row.isna().all() else row.idxmax(), axis=1
    #     )
    # )
    # Print the final string
    # print(result_string)
    # print('okay so its seeing the input correctly')

    for index, row in df_origin_cleaned.iterrows():
        df_origin_cleaned.at[index, 'Index'] = i
        df_target_cleaned.at[index, 'Index'] = i
        i += 1

    df_origin = df_origin_cleaned
    df_target = df_target_cleaned

    # grab all the conserved/very conserved position aas
    df_origin['conserved_positions'] = df_origin.apply(lambda row: [col for col in df_origin.columns if col != 'Index' 
                                                    and row[col] >= 0.2], axis=1)
    df_target['conserved_positions'] = df_target.apply(lambda row: [col for col in df_target.columns if col != 'Index' 
                                                        and row[col] >= 0.2], axis=1)

    df_origin['very_conserved_positions'] = df_origin.apply(lambda row: [col for col in df_origin.columns if col != 'Index' 
                                                            and col != 'conserved_positions' 
                                                            and row[col] >= 0.85], axis=1)
    df_target['very_conserved_positions'] = df_target.apply(lambda row: [col for col in df_target.columns if col != 'Index' 
                                                            and col != 'conserved_positions'
                                                            and row[col] >= 0.85], axis=1)

    # merge the dataframes with just the position/conservation values
    df_combined = pd.merge(
    df_origin[['Index', 'conserved_positions', 'very_conserved_positions']],
    df_target[['Index', 'conserved_positions', 'very_conserved_positions']],
    on='Index',
    suffixes=('_origin', '_target')
    )

    # now assign the priorities to each position
    df_combined['Priority'] = df_combined.apply(assign_priority, axis=1)

    # invert the priority to use it as a weight
    df_combined['Weight'] = 1 / df_combined['Priority']
    # then normalise the weights
    df_combined['Weight'] = df_combined['Weight'] / df_combined['Weight'].sum()

    # select order of mutation 'randomly' but using the probability weights
    # print('length of df', len(df_combined['Index']))
    mutation_order = df_combined.sample(n=len(df_combined), weights='Weight', replace=False)['Index'].tolist()

    # print(df_combined.sort_values(by='Priority', ascending=True))
    # print(df_combined['Index'])
    # print(mutation_order)

    return mutation_order


def get_grantham_distance(residue1, residue2):
    grantham_map = {
        'S':{'-':130, 'R':110, 'L':145, 'P':74, 'T':58, 'A':99 , 'V':124, 'G':56, 'I':142, 'F':155, 'Y':144, 'C':112, 'H':89, 'Q':68, 'N':46, 'K':121, 'D':65, 'E':80, 'M':135, 'W':177},
        'R':{'-':130, 'L':102, 'P':103, 'T':71, 'A':112, 'V':96, 'G':125, 'I':97, 'F':97, 'Y':77, 'C':180, 'H':29, 'Q':43, 'N':86, 'K':26, 'D':96, 'E':54, 'M':91, 'W':101},
        'L':{'-':130, 'P':98, 'T':92, 'A':96, 'V':32, 'G':138 ,'I':5, 'F':22, 'Y':36, 'C':198, 'H':99, 'Q':113, 'N':153, 'K':107, 'D':172, 'E':138, 'M':15, 'W':61},
        'P':{'-':130, 'T':38, 'A':27, 'V':68, 'G':42 ,'I':95, 'F':114, 'Y':110, 'C':169, 'H':77, 'Q':76, 'N':91, 'K':103, 'D':108, 'E':93, 'M':87, 'W':147},
        'T':{'-':130, 'A':58, 'V':69, 'G':59 ,'I':89, 'F':103, 'Y':92, 'C':149, 'H':47, 'Q':42, 'N':65, 'K':78, 'D':85, 'E':65, 'M':81, 'W':128},
        'A':{'-':130, 'V':64, 'G':60 ,'I':94, 'F':113, 'Y':112, 'C':195, 'H':86, 'Q':91, 'N':111, 'K':106, 'D':126, 'E':107, 'M':84, 'W':148}, 
        'V':{'-':130, 'G':109 ,'I':29, 'F':50, 'Y':55, 'C':192, 'H':84, 'Q':96, 'N':133, 'K':97, 'D':152, 'E':121, 'M':21, 'W':88},
        'G':{'-':130, 'I':135, 'F':153, 'Y':147, 'C':159, 'H':98, 'Q':87, 'N':80, 'K':127, 'D':94, 'E':98, 'M':127, 'W':184}, 
        'I':{'-':130, 'F':21, 'Y':33, 'C':198, 'H':94, 'Q':109, 'N':149, 'K':102, 'D':168, 'E':134, 'M':10, 'W':61}, 
        'F':{'-':130, 'Y':22, 'C':205, 'H':100, 'Q':116, 'N':158, 'K':102, 'D':177, 'E':140, 'M':28, 'W':40}, 
        'Y':{'-':130, 'C':194, 'H':83, 'Q':99, 'N':143, 'K':85, 'D':160, 'E':122, 'M':36, 'W':37},
        'C':{'-':130, 'H':174, 'Q':154, 'N':139, 'K':202, 'D':154, 'E':170, 'M':196, 'W':215}, 
        'H':{'-':130, 'Q':24, 'N':68, 'K':32, 'D':81, 'E':40, 'M':87, 'W':115}, 
        'Q':{'-':130, 'N':46, 'K':53, 'D':61, 'E':29, 'M':101, 'W':130}, 
        'N':{'-':130, 'K':94, 'D':23, 'E':42, 'M':142, 'W':174}, 
        'K':{'-':130, 'D':101, 'E':56, 'M':95, 'W':110}, 
        'D':{'-':130, 'E':45, 'M':160, 'W':181}, 
        'E':{'-':130, 'M':126, 'W':152},
        'M':{'-':130, 'W':67}
        }
    
    try:
        distance = grantham_map[residue1][residue2]
    except KeyError:
        distance = grantham_map[residue2][residue1]

    return distance

def get_grantham_mutations(seq1, seq2):
    # if the position is the same, ignore
    # if not, get distance

    # normalise distances to weights
    # select order by probability of weight
    # higher number means further away, so higher number = higher weight

    distances = []
    for position, (residue1, residue2) in enumerate(zip(seq1, seq2)):
        if residue1 != residue2:  # Only consider non-identical amino acids
            distance = get_grantham_distance(residue1, residue2)
            distances.append((position, residue2, distance))  # Store position, seq2 residue, and distance

    # calculate weightings based on distances
    max_distance = max(d for _, _, d in distances) if distances else 1  # Handle empty distances
    weightings = [(pos, res2, max_distance - d + 1) for pos, res2, d in distances]  # Add 1 to avoid zero weighting
    total_weight = sum(w for _, _, w in weightings)
    
    # normalize weights to create probabilities
    probabilities = [(pos, res2, w / total_weight) for pos, res2, w in weightings]

    # sample each position based on weighted probabilities
    positions, residues, prob_weights = zip(*probabilities)  # Separate components for sampling

    # Convert to numpy arrays for easier handling
    positions_array = np.array(list(zip(positions, residues)))
    probabilities = np.array(prob_weights)

    # Use numpy's choice function with replace=False to avoid getting duplicate mutations
    indices = np.random.choice(len(positions_array), size=len(prob_weights), replace=False, p=probabilities)
    sampled_positions = [(int(positions_array[i][0]), positions_array[i][1]) for i in indices]

    # this was giving replacements which was baaaaad
    # sampled_positions = random.choices(list(zip(positions, residues)), weights=prob_weights, k=len(prob_weights))

    return sampled_positions


def assign_consurf_priority(row):

    # basically if moving to highly conserved, that's higher priority
    # or if indel 

    origin = row['conservation_origin']
    target = row['conservation_target']

    if origin >= 8: #origin high conservation
        if target >= 8: #high to high, high priority
            return 1
        elif target >= 6: #high to low, mid priority
            return 5
        elif target > 0: #high to none, mid-low priority
            return 5
        else: #0 represents gap - high priority
            return 2
        
    elif origin >= 6:
        if target >= 8: #low to high, high priority
            return 2
        elif target >= 6: #low to low, mid priority
            return 6
        elif target > 0: #low to none, low priority
            return 7
        else: # low to gap - high-ish priority
            return 3
        
    elif origin > 0:
        if target >= 8: # none to high, high priority
            return 1
        elif target >= 6: #none to low, mid priority
            return 5
        elif target > 0: #none to none, very low priority
            return 9
        else: # none to gap - low priority
            return 2
        
    else:
        if target >= 8: #gap to high, higher priority
            return 1
        elif target >= 6: #gap to low, high priority
            return 3
        elif target > 0: #gap to none, mid-low priority
            return 8
        else: # gap to gap - not possible as matching position
            return 10


def calculate_consurf_mutation_positions(inputfile, removed_at):
    # expecting input datapath/dataset_direction_origin_target.extension
    dataset, originseq, targetseq = parse_filename(inputfile)

    datapath = '/'.join(inputfile.split('/')[:-1])
    originseq_file = f'{datapath}/{dataset}_{originseq}_consurf.csv'
    targetseq_file = f'{datapath}/{dataset}_{targetseq}_consurf.csv'

    df_origin = pd.read_csv(originseq_file, sep=',')
    df_target = pd.read_csv(targetseq_file, sep=',')

    # a workaround for having different indexes to the actual seq, just in case
    df_origin_cleaned = df_origin[~df_origin['index'].isin(removed_at)]
    df_target_cleaned = df_target[~df_target['index'].isin(removed_at)]

    # reset the indexing
    i = 0
    for index, row in df_origin_cleaned.iterrows():
        df_origin_cleaned.at[index, 'index'] = i
        df_target_cleaned.at[index, 'index'] = i
        i += 1

    df_origin = df_origin_cleaned
    df_target = df_target_cleaned

    # merge the dataframes for comparison
    df_merged = pd.merge(df_origin, df_target, on='index', suffixes=('_origin', '_target'))

    # remove rows where characters are the same
    df_merged = df_merged[df_merged['character_origin'] != df_merged['character_target']]

    df_merged['priority'] = df_merged.apply(assign_consurf_priority, axis=1)

    # invert the priority to use it as a weight
    df_merged['weight'] = 1 / df_merged['priority']
    # then normalise the weights
    df_merged['weight'] = df_merged['weight'] / df_merged['weight'].sum()

    # sample probabilistically from the possibilities to get mutation order
    mutation_order = df_merged.sample(n=len(df_merged), weights='weight', replace=False)['index'].tolist()

    return mutation_order



# ===============================================================================



def generate_mutations(inputfile, outputfile, mutation_position_output, method_type, positions=None, seed=42):
    """
    Assumes inputfile is fasta of two aligned sequences, 
    first is origin and second is target.

    outputfile is file to write mutated sequences to

    mutation_position_output is the file containing each sequence's mutated positions

    method_type is a string representing which mutation method to perform
    """


    random.seed(random.random())

    mutated_seqs = []

    records = list(SeqIO.parse(inputfile, 'fasta'))
    origin = records[0]
    target = records[1]

    if len(origin.seq) != len(target.seq):
        raise ValueError("These sequences are not aligned")


    # get the possible mutations for the specified method
    if method_type == 'random':
        # remove the gaps that are common (i.e. gaps in both sequences)
        origin, target, _ = remove_common_gaps(origin, target)

        possible_mutations = calculate_differences(origin, target)

        # shuffle the mutations
        random.shuffle(possible_mutations)


    elif method_type == 'specified':
        if positions is None:
            raise ValueError("positions were not specified")

        # remove the gaps that are common (i.e. gaps in both sequences)
        origin, target, _ = remove_common_gaps(origin, target)

        possible_mutations = get_specified_mutations(origin, target, positions)

        # shuffle the mutations
        random.shuffle(possible_mutations)


    elif method_type == 'nonconservative':
        # remove the gaps that are common (i.e. gaps in both sequences)
        origin, target, _ = remove_common_gaps(origin, target)

        possible_mutations = get_nonconservative_mutations(origin, target)

        # shuffle the mutations
        random.shuffle(possible_mutations)
        # print(f'noncon mutations: {possible_mutations}')

    elif method_type == 'grantham_distances':
        # remove the gaps that are common (i.e. gaps in both sequences)
        origin, target, _ = remove_common_gaps(origin, target)

        possible_mutations = get_grantham_mutations(origin, target)
        # possible_mutations = get_specified_mutations(origin, target, mutation_positions)
        # print(f'grantham mutations: {possible_mutations}')


    elif method_type == 'marginal_weights':
        origin, target, removed_at = remove_common_gaps(origin, target)
        # print(f'length origin {len(origin)}, length target {len(target)}')

        mutation_positions = get_probabilistic_mutations(inputfile, removed_at)
        possible_mutations = get_specified_mutations(origin, target, mutation_positions)

    elif method_type == 'ConSurf':
        origin, target, removed_at = remove_common_gaps(origin, target)

        mutation_positions = calculate_consurf_mutation_positions(inputfile, removed_at)
        possible_mutations = get_specified_mutations(origin, target, mutation_positions)


    
    # insert first (unmutated) sequence
    first = SeqRecord(
        Seq(str(origin.seq)),
        id=f"{origin.id}_{target.id}_0",
        description=''
    )

    # remove gaps
    record = remove_gaps(first)
    # record = first

    # store sequence as a mutable for alteration
    mutableseq = MutableSeq(str(origin.seq))

    mutated_seqs.append(record)

    i = 1

    cumulative_positions = []
    mutationfile = open(mutation_position_output, 'w')

    # for each mutation possible
    for pos, mutation in possible_mutations:

        # insert a blank line to represent no mutations for the first sequence
        # and to help format later sequences
        mutationfile.write('\n')

        # mutate the working sequence
        mutableseq[pos] = mutation

        # add the position to the cumulative list
        cumulative_positions.append(pos)

        # create a new seqrecord with the mutated sequence
        record = SeqRecord(
            Seq(str(mutableseq)),
            id=f"{origin.id}_{target.id}_{i}",
            description=''
        )

        record = remove_gaps(record)

        mutated_seqs.append(record) 
        
        # write the cumulative sequences to the tracking file
        # with open(mutation_position_output, 'w') as mutationfile:
        toprint = ','.join(str(pos) for pos in cumulative_positions)
        mutationfile.write(toprint)
            
        i += 1
        

    # write all sequences to output file
    SeqIO.write(mutated_seqs, outputfile, 'fasta')




# inputfile = snakemake.input
# outputfile = snakemake.output


# generate_mutations('../data/NR1_NR4_ancestors.fasta', '../data/testoutput.fasta')
# generate_mutations('../data/NR1_NR4_ancestors.fasta', '../data/testoutput.fasta', 'testposlist.txt', 
#                    'specified', [0,1,2,3,4])


# generate_mutations('../../data/reportdata/cd70_NR1toNR4_N6_N81.fasta', '../data/testoutput.fasta', 'testposlist.txt', 
#                    'marginal_weights')



# generate_mutations(snakemake.input.fasta, snakemake.output.fasta, snakemake.wildcards.method_name)

# this can take snakemake.wildcards.method_name as an extra input (make this the method type as a string)
# put this into the generate_mutations signature and do an if else statement for how to get the list of positions


generate_mutations(snakemake.input.fasta, 
                   snakemake.output.generated_sequences, 
                   snakemake.output.mutation_positions, 
                   snakemake.wildcards.method_name)

# generate_mutations('../data/reportdata/cd80_NR1toNR4_N7_N186.fasta','../data/testoutput.fasta','testposlist.txt',
#                    'ConSurf')

# generate_mutations('../data/reportdata/cd80_NR1toNR4_N7_N186.fasta','../data/testoutput.fasta','testposlist.txt',
#                    'grantham_distances')