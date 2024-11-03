from Bio import SeqIO
from Bio.Seq import Seq, MutableSeq
from Bio.SeqRecord import SeqRecord
import random

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


def pad_gaps(sequence, target_length):
    difference = abs(len(sequence) - target_length)
    # print(f'difference is {difference}')

    padded = ''
    gap_pad = '-' * difference

    padded += sequence.seq
    padded += gap_pad

    record = SeqRecord(
        Seq(padded),
        id=f"{sequence.id}",
        description=''
    )

    return record






def main():
    inputfile = snakemake.input.generated_sequences
    outputfile = snakemake.output.generated_sequences_padded

    records = list(SeqIO.parse(inputfile, 'fasta'))
    all_seqs = []

    longest_seq_length = max(len(records[0]), len(records[-1]))
    # print(f'first seq is {len(records[0])} long, second is {len(records[-1])}')
    for sequence in records:

        current = SeqRecord(
        Seq(str(sequence.seq)),
        id=f"{sequence.id}",
        description=''
    )
        
        # record = remove_gaps(current)
        record = pad_gaps(current, longest_seq_length)
        all_seqs.append(record)


    # write all sequences to output file
    SeqIO.write(all_seqs, outputfile, 'fasta')
    # print(f"all sequences: {all_seqs}")




if __name__ == "__main__":
    main()