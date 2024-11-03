import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import seq_utils
import os

def calculate_embeddings(sequence, model, tokenizer, model_type):
    inputs = tokenizer(" ".join(sequence), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        if model_type == "protbert":
            outputs = model(**inputs)
        elif model_type == "t5":
            outputs = model(**inputs.input_ids)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    embeddings = outputs.last_hidden_state

    # Mean pooling
    mean_embedding = embeddings.mean(dim=1).squeeze().numpy()

    # CLS token pooling
    cls_embedding = embeddings[:, 0].squeeze().numpy()

    # Max pooling
    max_embedding = embeddings.max(dim=1).values.squeeze().numpy()

    # Weighted pooling
    weights = torch.linspace(0.1, 1.0, embeddings.size(1), device=embeddings.device)
    weights = weights.unsqueeze(0).unsqueeze(-1)  # Add extra dimensions for broadcasting
    weighted_embedding = (embeddings * weights).mean(dim=1).squeeze().numpy()

    return {
        f"{model_type}_mean_embedding": mean_embedding,
        f"{model_type}_cls_embedding": cls_embedding,
        f"{model_type}_max_embedding": max_embedding,
        f"{model_type}_weighted_embedding": weighted_embedding,
    }



def parse_seq_info(info):
    split_info = info.split('_')
    mutation_no = split_info[-1]

    return mutation_no



def process_and_store_embeddings(df, model_name, embedding_df_path, model_type):

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load existing embeddings if they exist
    if os.path.exists(embedding_df_path):
        embedding_df = pd.read_pickle(embedding_df_path)
    else:
        try:
            embedding_df = pd.DataFrame(columns=["info", "sequence", "model_name"], dtype=object)
        except Exception as e:
            print(f"Error initializing DataFrame: {e}")
            raise
    # print ('now done')

    for idx, row in df.iterrows():
        info = row["info"]
        sequence = row["sequence"]

        existing_row = embedding_df[
            (embedding_df["info"] == info) & (embedding_df["model_name"] == model_name)
        ]

        if not existing_row.empty and f"{model_type}_mean_embedding" in existing_row.columns:
            # Ensure the specific column has data
            if not existing_row[f"{model_type}_mean_embedding"].empty:
                continue  # Skip if embeddings for this sequence already exist

        try:
            embeddings = calculate_embeddings(sequence, model, tokenizer, model_type)
            new_row = {
                "info": info,
                "sequence": sequence,
                "model_name": model_name,
                **embeddings,
            }
            embedding_df = pd.concat([embedding_df, pd.DataFrame([new_row])], ignore_index=True)

        except Exception as e:
            print(f"Failed to process sequence {sequence} with error: {e}")

    # add the no of mutations to sequence to dataframe
    # needed in order to plot correctly
    embedding_df['num_mutation'] = embedding_df['info'].apply(parse_seq_info)


    # Save embedding_df with full embeddings
    embedding_df.to_pickle(embedding_df_path)
    merged_df = pd.merge(df, embedding_df, on=['info', 'sequence'], how='left')

    return merged_df


def main():
    # df = seq_utils.get_sequence_df(snakemake.input.generated_sequences_padded, alignment=True)
    # df = seq_utils.get_sequence_df(snakemake.input.generated_sequences_padded)
    df = seq_utils.get_sequence_df(snakemake.input.input_sequences)


    # Set model name and output paths
    bert_model_name = "yarongef/DistilProtBert"
    bert_embedding_df_path = snakemake.output.embedding_df

    # Process and store embeddings
    embedding_df = process_and_store_embeddings(df, bert_model_name, bert_embedding_df_path, model_type='protbert')


if __name__ == "__main__":
    main()
