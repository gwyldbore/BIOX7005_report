import pandas as pd
from Bio import SeqIO, AlignIO

def get_sequence_df(
    *fasta_paths,
    drop_duplicates=True,
    alignment=False,
    ancestor=False,
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ-",
):
    seq_list = []
    duplicates = {}

    cols = [
        "info",
        "truncated_info",
        "extracted_id",
        "extracted_name",
        "sequence",
        "original_fasta",
    ]

    if alignment or ancestor:
        print("This is an alignment")
        cols.append("original_alignment")
        cols.append("Sequence_aligned")

    # if ancestor:
    #     cols.append("Sequence_aligned")

    # print('reached here')
    for fasta_path in fasta_paths:
        # print('then also reached here')
        # Load FASTA file
        # seqs = sequence.readFastaFile(fasta_path, alpha)

        if alignment:
            # print('then reached the alignment line')
            seqs = AlignIO.parse(open(fasta_path), format="fasta")
            # print('and the opening didnt fail')

        else:
            # print('or the else line')
            seqs = SeqIO.parse(open(fasta_path), format="fasta")

        # print('so then entering seq in seqs loop')
        # print(f'seqs: {seqs}')
        # Add to annotation file
        for seq in seqs:
            if alignment == False:
                if seq.name in duplicates:
                    print(
                        f"DUPLICATE:{seq.name} is in {duplicates[seq.name]} and {fasta_path}\n"
                    )
                else:
                    duplicates[seq.name] = fasta_path

                curr_seq = [
                    seq.id,
                    seq.id.split(" ")[0],
                    seq.id.split("|")[1]
                    if len(seq.id.split("|")) > 1
                    else seq.id.split(" ")[0],
                    seq.id.split("|")[-1],
                    "".join(str(seq.seq).replace("-", ""))
                    if len(seq.seq) > 0
                    else None,
                    fasta_path,
                ]

                seq_list.append(curr_seq)

            elif alignment:
                for aligned_seq in seq:
                    curr_seq = [
                        aligned_seq.id,
                        aligned_seq.id.split(" ")[0],
                        aligned_seq.id.split("|")[1]
                        if len(aligned_seq.id.split("|")) > 1
                        else aligned_seq.id.split(" ")[0],
                        aligned_seq.id.split("|")[-1],
                        "".join(str(aligned_seq.seq).replace("-", ""))
                        if len(aligned_seq.seq) > 0
                        else None,
                        None,
                        fasta_path,
                        "".join(aligned_seq.seq),
                    ]
                    seq_list.append(curr_seq)

            # if ancestor:
            #     curr_seq.append("".join(aligned_seq.seq))

    df = pd.DataFrame(seq_list, columns=cols)

    if drop_duplicates:
        df = df.drop_duplicates(subset="info", keep="first")

    # Drop the sequence column if there are no sequences (i.e. if we just added a list of identifiers)
    nan_value = float("NaN")

    # df.replace("", nan_value, inplace=True)

    df.dropna(how="all", axis=1, inplace=True)

    return df


def tag_node(info, dataset):
    with open('./data/NR1_ids.txt', 'r') as file:
        nr1_names = set(line.strip() for line in file)

    # with open(f'./data/reportdata/{dataset}_nodes/NR1_ids.txt', 'r') as file:
    #     nr1_names = set(line.strip() for line in file)

    with open('./data/NR4_ids.txt', 'r') as file:
        nr4_names = set(line.strip() for line in file)

    # with open(f'./data/reportdata/{dataset}_nodes/NR4_ids.txt', 'r') as file:
    #     nr4_names = set(line.strip() for line in file)

    if info in nr1_names:
        return 'NR1'
    elif info in nr4_names:
        return 'NR4'
    else:
        return 'Other'