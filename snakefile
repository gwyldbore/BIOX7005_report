try:
    NUM_REPLICATES = config['num_replicates']
except KeyError:
    NUM_REPLICATES = 50


# actual path
try:
    DATASET_PATH = config['dataset_path']
except KeyError:
    DATASET_PATH = 'data/reportdata'

try:
    ALL_DATASET_NAMES = config['all_dataset_names']
except KeyError: # relevant dataset just needs to be uncommented
    ALL_DATASET_NAMES = ['cd70']
    # ALL_DATASET_NAMES = ['cd80']
    # ALL_DATASET_NAMES = ['cd85']

try:
    ALL_DATAFILE_NAMES = config['all_datafile_names']
except KeyError:
    ALL_DATAFILE_NAMES = ['NR1toNR4_N6_N81', 'NR4toNR1_N81_N6']
    # ALL_DATAFILE_NAMES = ['NR1toNR4_N7_N186', 'NR4toNR1_N186_N7'] 
    # ALL_DATAFILE_NAMES = ['NR1toNR4_N7_N299', 'NR4toNR1_N299_N7']

try:
    INTERPROSCAN_DIR = config['interproscan_dir']
except KeyError:
    INTERPROSCAN_DIR = '/media/WorkingSpace/Share/interproscan/interproscan-5.65-97.0'

try:
    METHOD_NAMES = config['method_names']
except KeyError:
    METHOD_NAMES = ['random', 'nonconservative', 'grantham_distances', 'marginal_weights', 'ConSurf']




rule all:
    input:
        merged_df= expand("workflows/{dataset_name}/{method_name}/merged/{datafile_name}_{rep}.csv",
        dataset_name=ALL_DATASET_NAMES,
        datafile_name=ALL_DATAFILE_NAMES,
        method_name=METHOD_NAMES,
        rep=range(1, NUM_REPLICATES + 1),
        ), 

        statsplot= expand("workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_{graph_type}_{method_name}.png",
        dataset_name=ALL_DATASET_NAMES,
        datafile_name=ALL_DATAFILE_NAMES,
        method_name=METHOD_NAMES,
        graph_type=['mutation', 'position']
        ),

        single_pca= expand("workflows/{dataset_name}/{method_name}/plots/{dataset_name}_{datafile_name}_{rep}_mutations_{method_name}.png",
        dataset_name=ALL_DATASET_NAMES,
        datafile_name=ALL_DATAFILE_NAMES,
        method_name=METHOD_NAMES,
        rep=range(1, NUM_REPLICATES + 1, 10),
        ),

        calculate_stats= expand("workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_boxplot_first.png",
        dataset_name=ALL_DATASET_NAMES,
        datafile_name=ALL_DATAFILE_NAMES,
        ),

        grouped_pca = expand("workflows/{dataset_name}/{method_name}/plots/{dataset_name}_{datafile_name}_{batch}_{method_name}_mutations_grouped.png",
        dataset_name=ALL_DATASET_NAMES,
        datafile_name=ALL_DATAFILE_NAMES,
        method_name=METHOD_NAMES,
        batch=range(1, (NUM_REPLICATES + 4) // 5),
        ),







# rule generate_ancestor_embeddings:
#     input:
#         # input_sequences="data/reportdata/combined_nodes/combined_ancestors.fa"
#         input_sequences=DATASET_PATH + "/{dataset_name}_ancestors.fa"
#     output:
#         # embedding_df="data/reportdata/ancestor_embedding_combined_df.csv"
#         embedding_df="workflows/{dataset_name}/embeddings/ancestor_embedding_df.csv"
#         # embedding_df="workflows/{dataset_name}/reportdata/ancestor_embedding_combined_df.csv"
#     script:
#         "scripts/generate_ancestor_embeddings.py"


rule generate_mutations:
    input:
        fasta=DATASET_PATH + "/{dataset_name}_{datafile_name}.fasta"
    output:
        generated_sequences="workflows/{dataset_name}/{method_name}/fasta/ungapped/{datafile_name}_{rep}.fasta",
        mutation_positions="workflows/{dataset_name}/{method_name}/fasta/{datafile_name}_{rep}_mutationpos.txt"
    script:
        "scripts/generate_mutations.py"


rule train_logistic_regression:
    input:
        ancestor_embeddings="data/ancestor_embedding_df.csv"
    output:
        # model_output="workflows/{dataset_name}/logregmodel/trained_logreg.pkl"
        model_output="workflows/logregmodel/trained_logreg.pkl"
    script:
        "scripts/train_logistic_regression.py"


rule predict_logistic_regression:
    input:
        model="workflows/logregmodel/trained_logreg.pkl",
        embedding_df="workflows/{dataset_name}/{method_name}/embeddings/{datafile_name}_{rep}.csv",
    output:
        logreg_results="workflows/{dataset_name}/{method_name}/logregprediction/{datafile_name}_{rep}.csv"
    script:
        "scripts/predict_logistic_regression.py"


rule pad_with_gaps:
    input:
        generated_sequences="workflows/{dataset_name}/{method_name}/fasta/ungapped/{datafile_name}_{rep}.fasta"
    output:
        generated_sequences_padded="workflows/{dataset_name}/{method_name}/fasta/padded/{datafile_name}_{rep}.fasta"
    script:
        "scripts/remove_gaps.py"


rule run_interproscan:
    input:
        generated_sequences="workflows/{dataset_name}/{method_name}/fasta/ungapped/{datafile_name}_{rep}.fasta"
    output:
        interproscan_df="workflows/{dataset_name}/{method_name}/interproscan/{datafile_name}_{rep}.csv"
    params:
        interproscan_dir=INTERPROSCAN_DIR
    shell:
        """
        {params.interproscan_dir}/interproscan.sh \
        -dp \
        -i {input.generated_sequences} \
        -appl Gene3D,PRINTS \
        -f tsv -o {output.interproscan_df}
        """




rule generate_embeddings:
    input:
        input_sequences="workflows/{dataset_name}/{method_name}/fasta/padded/{datafile_name}_{rep}.fasta",
    output:
        embedding_df="workflows/{dataset_name}/{method_name}/embeddings/{datafile_name}_{rep}.csv",
    script:
        "scripts/generate_embeddings.py"

rule run_blast:
    input:
        generated_sequences="workflows/{dataset_name}/{method_name}/fasta/ungapped/{datafile_name}_{rep}.fasta",
    output:
        blast_out="workflows/{dataset_name}/{method_name}/blast/blastresults_{datafile_name}_{rep}.tsv",
    shell:
        "blastp -db ../BLAST/NR_blastdb -query {input.generated_sequences} -out {output.blast_out} -evalue 1e-50 -num_threads 4 -max_target_seqs 10000 -outfmt '7 qseqid salltitles'"

rule parse_blast:
    input:
        generated_sequences="workflows/{dataset_name}/{method_name}/fasta/ungapped/{datafile_name}_{rep}.fasta",
        blast_out="workflows/{dataset_name}/{method_name}/blast/blastresults_{datafile_name}_{rep}.tsv",
    output:
        blast_df="workflows/{dataset_name}/{method_name}/blast/{datafile_name}_{rep}.csv",
    script:
        "scripts/parse_blast.py"

rule merge_outputs:
    input:
        interproscan_df="workflows/{dataset_name}/{method_name}/interproscan/{datafile_name}_{rep}.csv",
        embedding_df="workflows/{dataset_name}/{method_name}/embeddings/{datafile_name}_{rep}.csv",
        blast_df="workflows/{dataset_name}/{method_name}/blast/{datafile_name}_{rep}.csv",
        mutationfile="workflows/{dataset_name}/{method_name}/fasta/{datafile_name}_{rep}_mutationpos.txt",
        logreg_results="workflows/{dataset_name}/{method_name}/logregprediction/{datafile_name}_{rep}.csv"

    output:
        merged_df="workflows/{dataset_name}/{method_name}/merged/{datafile_name}_{rep}.csv",
    script:
        "scripts/merge_outputs.py"

rule visualise_pca:
    input:
        embedding_df="workflows/{dataset_name}/{method_name}/embeddings/{datafile_name}_{rep}.csv",
        ancestor_embeddings="data/ancestor_embedding_df.csv",
        predictions_df="workflows/{dataset_name}/{method_name}/results/{datafile_name}_{rep}.csv"
    output:
        plot_mutation="workflows/{dataset_name}/{method_name}/plots/{dataset_name}_{datafile_name}_{rep}_mutations_{method_name}.png",
        plot_prediction="workflows/{dataset_name}/{method_name}/plots/{dataset_name}_{datafile_name}_{rep}_predictions_{method_name}.png"
    script:
        "scripts/plot_pca.py"

rule grouped_pca:
    input:
        embedding_df=lambda wildcards: expand(
            "workflows/{dataset_name}/{method_name}/embeddings/{datafile_name}_{rep}.csv", 
            dataset_name=wildcards.dataset_name,
            method_name=wildcards.method_name,
            datafile_name=wildcards.datafile_name,
            rep=range(int(wildcards.batch) * 5, (int(wildcards.batch) + 1) * 5)
        ),
        predictions_df=lambda wildcards: expand(
            "workflows/{dataset_name}/{method_name}/results/{datafile_name}_{rep}.csv", 
            dataset_name=wildcards.dataset_name,
            method_name=wildcards.method_name,
            datafile_name=wildcards.datafile_name,
            rep=range(int(wildcards.batch) * 5, (int(wildcards.batch) + 1) * 5)
        ),
        ancestor_embeddings="data/ancestor_embedding_df.csv",

    output:
        plot_mutation="workflows/{dataset_name}/{method_name}/plots/{dataset_name}_{datafile_name}_{batch}_{method_name}_mutations_grouped.png",
        plot_prediction="workflows/{dataset_name}/{method_name}/plots/{dataset_name}_{datafile_name}_{batch}_{method_name}_predictions_grouped.png"
    script:
        "scripts/plot_pca_grouped.py"



rule gather_prediction_info:
    input:
        merged_df="workflows/{dataset_name}/{method_name}/merged/{datafile_name}_{rep}.csv"
    output:
        results_df="workflows/{dataset_name}/{method_name}/results/{datafile_name}_{rep}.csv",
    script:
        "scripts/gather_prediction_info.py"



rule perform_statistical_testing:
    input:
        lambda wildcards: expand(
            "workflows/{dataset_name}/{method_name}/results/{datafile_name}_{rep}.csv",
            dataset_name=wildcards.dataset_name,
            datafile_name=wildcards.datafile_name,
            method_name=METHOD_NAMES,
            rep=range(1, NUM_REPLICATES + 1)
        )
    output:
        boxplot_first="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_boxplot_first.png",
        first_df="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_firstchanges.csv",
        qqplot_first="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_qqplot_first.png",
        shapiro_first="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_shapiro_first.txt",
        kruskal_first="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_kruskal_first.txt",
        multi_df="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_multichanges.csv",
        boxplot_multi="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_boxplot_multi.png",
        qqplot="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_qqplot_grouped.png",
        shapiro="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_shapiro_grouped.txt",
        kruskal="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_kruskal_grouped.txt",
        boxplot_combined="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_boxplot_combined.png"
    script:
        "scripts/statistical_testing.py"



rule plot_summary_statistics:
    input:
        lambda wildcards: expand(
            "workflows/{dataset_name}/{method_name}/results/{datafile_name}_{rep}.csv",
            dataset_name=wildcards.dataset_name,
            method_name=wildcards.method_name,
            datafile_name=wildcards.datafile_name,
            rep=range(1, NUM_REPLICATES + 1)
        )
    output:
        mutation_graphs="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_mutation_{method_name}.png",
        position_graphs="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_position_{method_name}.png",
        mutation_spread="workflows/{dataset_name}/results/{dataset_name}_{datafile_name}_mutationspread_{method_name}.png",
    script:
        "scripts/create_stats_plots.py"
