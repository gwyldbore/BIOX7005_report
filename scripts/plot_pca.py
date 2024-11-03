import pickle
import pandas as pd
import seq_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def plot_pca(all_embeddings_df, nodes_to_label, outpath, col_name='protbert_cls_embedding'):

    embeddings = np.vstack(all_embeddings_df[col_name].values)


    # Apply PCA
    num_components = 2
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(embeddings)

    # Add PCA results to the DataFrame
    all_embeddings_df['pca1'] = pca_result[:, 0]
    all_embeddings_df['pca2'] = pca_result[:, 1]

    # Get unique clades
    clades_with_color = all_embeddings_df['Clade'].dropna().unique()
    num_clades = len(clades_with_color)

    # Define color map for the clades
    # colors = plt.cm.get_cmap('Set1', num_clades).colors
    # colors = plt.colormaps['PiYG'].resampled(num_clades)
    # clade_cmap = ListedColormap(['#d3d3d3', '#a9a9a9'])

    clade_cmap = ListedColormap(['#d3d3d3', '#ffca8a'])
    colors = plt.get_cmap(clade_cmap, num_clades).colors

    # plt.figure(figsize=(20, 14))
    fig, ax = plt.subplots(figsize=(20,14))

    # Plot all points in gray first to show entries with no clade
    no_clade_df = all_embeddings_df[all_embeddings_df['Clade'].isna()]
    # probably a way to set colour=blue to be a gradient??
    # plt.scatter(no_clade_df['pca1'], no_clade_df['pca2'], color='blue', alpha=0.5, label='No Clade')
    ax.scatter(no_clade_df['pca1'], no_clade_df['pca2'], color='blue', alpha=0.5, label='No Clade')

    # Plot points with clades in different colors
    for clade, color in zip(clades_with_color, colors):
        subset = all_embeddings_df[all_embeddings_df['Clade'] == clade]
        # plt.scatter(subset['pca1'], subset['pca2'], label=clade, color=color)
        ax.scatter(subset['pca1'], subset['pca2'], label=clade, color=color)


    mutation_df = all_embeddings_df.dropna(subset=['num_mutation'])

    # viridis = plt.get_cmap('viridis')
    # new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    #     f"trunc({'viridis'},{0.2},{0.8})",
    #     viridis(np.linspace(0.2, 0.8, 256))
    # )

    # scatter = plt.scatter(mutation_df['pca1'], mutation_df['pca2'], 
    #             c=[int(x) for x in mutation_df['num_mutation']], cmap='cool')
    scatter = ax.scatter(mutation_df['pca1'], mutation_df['pca2'], 
                c=[int(x) for x in mutation_df['num_mutation']], cmap='cool')
    
    cax = ax.inset_axes([0.05, 0.05, 0.3, 0.05])
    fig.colorbar(scatter, cax=cax, orientation='horizontal')
    
    # cbar = plt.colorbar()
    # cbar_ax = fig.add_subplot(gs[1])
    # fig.colorbar(scatter, cax=cbar_ax, orientation='vertical')
    # cbar_ax.set_ylabel('Colorbar')

    # Set plot titles and labels
    plt.title("PCA by Clade")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(outpath)

def plot_pca_ancestors_static(mutations_df, ancestors_df, nodes_to_label, outpath, col_name='protbert_cls_embedding'):
    
    ancestor_embeddings = np.vstack(ancestors_df[col_name].values)

    # pca = PCA(n_components=2)
    pca = PCA(n_components=15)
    pca_result = pca.fit(ancestor_embeddings)

    explained_variance_ratio = pca.explained_variance_ratio_

    # Plot the variance explained by each component
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center')
    plt.step(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), where='mid', color='orange')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Variance Explained by Each Principal Component')
    plt.savefig('PCA_explained.png')


    # Transform both ancestors_df and mutations_df using the fitted PCA
    ancestors_df[['pca1', 'pca2']] = pca.transform(ancestor_embeddings)[:, :2]
    mutations_df[['pca1', 'pca2']] = pca.transform(np.vstack(mutations_df[col_name].values))[:, :2]


    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 14))

    # Plot mutations_df entries in blue (No Clade)
    # ax.scatter(mutations_df['pca1'], mutations_df['pca2'], color='blue', alpha=0.5)

    # Plot entries from ancestors_df with clades in different colors
    clades_with_color = ancestors_df['Clade'].unique()
    num_clades = len(clades_with_color)

    clade_cmap = ListedColormap(['#d3d3d3', '#ffca8a', '#99ff8a'])
    colors = plt.get_cmap(clade_cmap, num_clades).colors

    # Plot points with clades in different colors
    for clade, color in zip(clades_with_color, colors):
        subset = ancestors_df[ancestors_df['Clade'] == clade]
        ax.scatter(subset['pca1'], subset['pca2'], label=clade, color=color)


    mutation_df = mutations_df.dropna(subset=['num_mutation'])
    scatter = ax.scatter(mutation_df['pca1'], mutation_df['pca2'], 
                c=[int(x) for x in mutation_df['num_mutation']], cmap='cool')
    
    cax = ax.inset_axes([0.05, 0.05, 0.3, 0.05])
    fig.colorbar(scatter, cax=cax, orientation='horizontal')

    # Set plot titles and labels
    plt.title("PCA by Clade")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(outpath)



def plot_pca_colour_by_predicted(all_embeddings_df, nodes_to_label, outpath, col_name='protbert_cls_embedding'):

    embeddings = np.vstack(all_embeddings_df[col_name].values)


    # Apply PCA
    num_components = 2
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(embeddings)

    # Add PCA results to the DataFrame
    all_embeddings_df['pca1'] = pca_result[:, 0]
    all_embeddings_df['pca2'] = pca_result[:, 1]

    # Get unique clades
    clades_with_color = all_embeddings_df['Clade'].dropna().unique()
    num_clades = len(clades_with_color)

    # Define color map for the clades
    # colors = plt.cm.get_cmap('Set1', num_clades).colors
    # clade_cmap = ListedColormap(['royalblue', 'green'])
    clade_cmap = ListedColormap(['#d3d3d3', '#ffca8a'])
    colors = plt.get_cmap(clade_cmap, num_clades).colors

    plt.figure(figsize=(20, 14))
    # fig, ax = plt.subplots(figsize=(20, 14))

    # Plot all points in gray first to show entries with no clade
    no_clade_df = all_embeddings_df[all_embeddings_df['Clade'].isna()]
    # probably a way to set colour=blue to be a gradient??
    plt.scatter(no_clade_df['pca1'], no_clade_df['pca2'], color='blue', alpha=0.5, label='No Clade')

    # Plot points with clades in different colors
    for clade, color in zip(clades_with_color, colors):
        subset = all_embeddings_df[all_embeddings_df['Clade'] == clade]
        plt.scatter(subset['pca1'], subset['pca2'], label=f'Clade: {clade}', color=color)
        # ax.scatter(subset['pca1'], subset['pca2'], label=f'Clade: {clade}', color=color)



    # Overlay predictions with new colors
    prediction_df = all_embeddings_df.dropna(subset=['overall_prediction'])
    unique_predictions = prediction_df['overall_prediction'].unique()
    unique_predictions.sorted()
    # print(unique_predictions)

    # Define a new colormap for predictions
    prediction_cmap = ListedColormap(['mediumorchid', 'red', 'royalblue', 'forestgreen'])
    # if unique_predictions[0] == 'NR1':
    #     prediction_cmap = ListedColormap(['mediumorchid', 'red', 'royalblue', 'forestgreen'])
    # else:
    #     prediction_cmap = ListedColormap(['forestgreen', 'royalblue', 'red', 'mediumorchid'])
        

    prediction_colors = plt.get_cmap(prediction_cmap).colors

    for prediction, color in zip(unique_predictions, prediction_colors):
        pred_subset = prediction_df[prediction_df['overall_prediction'] == prediction]
        plt.scatter(pred_subset['pca1'], pred_subset['pca2'], 
                    color=color, label=f'Prediction: {prediction}')

    # Set plot titles and labels
    plt.title("PCA by Clade")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(outpath)

def plot_pca_colour_by_predicted_ancestors_static(mutations_df, ancestors_df, nodes_to_label, outpath, col_name='protbert_cls_embedding'):
    ancestor_embeddings = np.vstack(ancestors_df[col_name].values)

    pca = PCA(n_components=2)
    pca_result = pca.fit(ancestor_embeddings)


    # Transform both ancestors_df and mutations_df using the fitted PCA
    ancestors_df[['pca1', 'pca2']] = pca.transform(ancestor_embeddings)
    mutations_df[['pca1', 'pca2']] = pca.transform(np.vstack(mutations_df[col_name].values))


    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 14))

    # Plot mutations_df entries in blue (No Clade)
    # ax.scatter(mutations_df['pca1'], mutations_df['pca2'], color='blue', alpha=0.5)

    # Plot entries from ancestors_df with clades in different colors
    clades_with_color = ancestors_df['Clade'].unique()
    num_clades = len(clades_with_color)

    clade_cmap = ListedColormap(['#d3d3d3', '#ffca8a'])
    colors = plt.get_cmap(clade_cmap, num_clades).colors

    # Plot points with clades in different colors
    for clade, color in zip(clades_with_color, colors):
        subset = ancestors_df[ancestors_df['Clade'] == clade]
        ax.scatter(subset['pca1'], subset['pca2'], label=clade, color=color)


    # Overlay predictions with new colors
    prediction_df = mutations_df.dropna(subset=['overall_prediction'])
    # unique_predictions = prediction_df['overall_prediction'].unique()
    unique_predictions = ['NR1', 'NR1-like', 'NR4-like', 'NR4', 'other']

    # define new cmap 
    prediction_cmap = ListedColormap(['darkorchid', 'forestgreen', 'magenta', 'firebrick',  'royalblue'])
    if unique_predictions[0] == 'NR4':
        prediction_cmap = ListedColormap(['royalblue', 'firebrick', 'magenta', 'forestgreen', 'darkorchid'])


    prediction_colors = plt.get_cmap(prediction_cmap, len(unique_predictions)).colors

    for prediction, color in zip(unique_predictions, prediction_colors):
        pred_subset = prediction_df[prediction_df['overall_prediction'] == prediction]
        plt.scatter(pred_subset['pca1'], pred_subset['pca2'], 
                    color=color, label=f'Prediction: {prediction}', alpha=0.8)

    # Set plot titles and labels
    plt.title("PCA by Clade")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(outpath)



def plot_tsne_ancestors_static(mutations_df, ancestors_df, nodes_to_label, outpath, col_name='protbert_cls_embedding', perplexity=30, n_iter=1000):
    
    # Stack ancestor embeddings and perform t-SNE
    ancestor_embeddings = np.vstack(ancestors_df[col_name].values)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_result_ancestors = tsne.fit_transform(ancestor_embeddings)
    
    # Add the t-SNE results back to ancestors_df
    ancestors_df[['tsne1', 'tsne2']] = tsne_result_ancestors

    # Perform t-SNE on mutations_df
    mutation_embeddings = np.vstack(mutations_df[col_name].values)
    tsne_result_mutations = tsne.fit_transform(mutation_embeddings)
    mutations_df[['tsne1', 'tsne2']] = tsne_result_mutations

    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 14))

    # Plot entries from ancestors_df with clades in different colors
    clades_with_color = ancestors_df['Clade'].unique()
    num_clades = len(clades_with_color)

    # Define colors for clades
    clade_cmap = ListedColormap(['#d3d3d3', '#ffca8a', '#99ff8a'])
    colors = plt.get_cmap(clade_cmap, num_clades).colors

    for clade, color in zip(clades_with_color, colors):
        subset = ancestors_df[ancestors_df['Clade'] == clade]
        ax.scatter(subset['tsne1'], subset['tsne2'], label=clade, color=color, alpha=0.7, s=50)

    # Plot mutations with color based on 'num_mutation'
    mutation_df = mutations_df.dropna(subset=['num_mutation'])
    scatter = ax.scatter(mutation_df['tsne1'], mutation_df['tsne2'], 
                         c=[int(x) for x in mutation_df['num_mutation']], cmap='cool', alpha=0.6, s=50)
    
    # Add colorbar for mutation count
    cax = ax.inset_axes([0.05, 0.05, 0.3, 0.05])
    fig.colorbar(scatter, cax=cax, orientation='horizontal')

    # Set plot titles and labels
    plt.title("t-SNE by Clade")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig(outpath)

def plot_tsne_colour_by_predicted_ancestors_static(mutations_df, ancestors_df, nodes_to_label, outpath, col_name='protbert_cls_embedding', perplexity=30, n_iter=1000):
    # Stack ancestor embeddings and perform t-SNE
    ancestor_embeddings = np.vstack(ancestors_df[col_name].values)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_result_ancestors = tsne.fit_transform(ancestor_embeddings)
    
    # Add the t-SNE results back to ancestors_df
    ancestors_df[['tsne1', 'tsne2']] = tsne_result_ancestors

    # Perform t-SNE on mutations_df
    mutation_embeddings = np.vstack(mutations_df[col_name].values)
    tsne_result_mutations = tsne.fit_transform(mutation_embeddings)
    mutations_df[['tsne1', 'tsne2']] = tsne_result_mutations

    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 14))

    # Plot entries from ancestors_df with clades in different colors
    clades_with_color = ancestors_df['Clade'].unique()
    num_clades = len(clades_with_color)

    # Define colors for clades
    clade_cmap = ListedColormap(['#d3d3d3', '#ffca8a', '#99ff8a'])
    colors = plt.get_cmap(clade_cmap, num_clades).colors

    for clade, color in zip(clades_with_color, colors):
        subset = ancestors_df[ancestors_df['Clade'] == clade]
        ax.scatter(subset['tsne1'], subset['tsne2'], label=clade, color=color, alpha=0.7, s=50)

    # Overlay predictions with new colors
    prediction_df = mutations_df.dropna(subset=['overall_prediction'])
    unique_predictions = prediction_df['overall_prediction'].unique()

    # Define new cmap for predictions
    prediction_cmap = ListedColormap(['mediumorchid', 'red', 'royalblue', 'forestgreen'])
    if unique_predictions[0] == 'NR4':
        prediction_cmap = ListedColormap(['forestgreen', 'royalblue', 'red', 'mediumorchid'])

    prediction_colors = plt.get_cmap(prediction_cmap, len(unique_predictions)).colors

    for prediction, color in zip(unique_predictions, prediction_colors):
        pred_subset = prediction_df[prediction_df['overall_prediction'] == prediction]
        plt.scatter(pred_subset['tsne1'], pred_subset['tsne2'], 
                    color=color, label=f'Prediction: {prediction}', alpha=0.6, s=50)

    # Set plot titles and labels
    plt.title("t-SNE by Clade")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig(outpath)


def plot_umap_ancestors_static(mutations_df, ancestors_df, nodes_to_label, outpath, col_name='protbert_cls_embedding', n_neighbors=15, min_dist=0.1):
    # Stack ancestor embeddings and perform UMAP
    ancestor_embeddings = np.vstack(ancestors_df[col_name].values)
    umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_result_ancestors = umap_model.fit_transform(ancestor_embeddings)

    # Add the UMAP results back to ancestors_df
    ancestors_df[['umap1', 'umap2']] = umap_result_ancestors

    # Perform UMAP on mutations_df
    mutation_embeddings = np.vstack(mutations_df[col_name].values)
    umap_result_mutations = umap_model.fit_transform(mutation_embeddings)
    mutations_df[['umap1', 'umap2']] = umap_result_mutations

    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 14))

    # Plot entries from ancestors_df with clades in different colors
    clades_with_color = ancestors_df['Clade'].unique()
    num_clades = len(clades_with_color)

    clade_cmap = ListedColormap(['#d3d3d3', '#ffca8a', '#99ff8a'])
    colors = plt.get_cmap(clade_cmap, num_clades).colors

    for clade, color in zip(clades_with_color, colors):
        subset = ancestors_df[ancestors_df['Clade'] == clade]
        ax.scatter(subset['umap1'], subset['umap2'], label=clade, color=color, alpha=0.7, s=50)

    # Plot mutations with color based on 'num_mutation'
    mutation_df = mutations_df.dropna(subset=['num_mutation'])
    scatter = ax.scatter(mutation_df['umap1'], mutation_df['umap2'], 
                         c=[int(x) for x in mutation_df['num_mutation']], cmap='cool', alpha=0.6, s=50)

    # Add colorbar for mutation count
    cax = ax.inset_axes([0.05, 0.05, 0.3, 0.05])
    fig.colorbar(scatter, cax=cax, orientation='horizontal')

    # Set plot titles and labels
    plt.title("UMAP by Clade")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.savefig(outpath)
def plot_umap_colour_by_predicted_ancestors_static(mutations_df, ancestors_df, nodes_to_label, outpath, col_name='protbert_cls_embedding', n_neighbors=15, min_dist=0.1):
    # Stack ancestor embeddings and perform UMAP
    ancestor_embeddings = np.vstack(ancestors_df[col_name].values)
    umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_result_ancestors = umap_model.fit_transform(ancestor_embeddings)
    
    # Add the UMAP results back to ancestors_df
    ancestors_df[['umap1', 'umap2']] = umap_result_ancestors

    # Perform UMAP on mutations_df
    mutation_embeddings = np.vstack(mutations_df[col_name].values)
    umap_result_mutations = umap_model.fit_transform(mutation_embeddings)
    mutations_df[['umap1', 'umap2']] = umap_result_mutations

    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 14))

    # Plot entries from ancestors_df with clades in different colors
    clades_with_color = ancestors_df['Clade'].unique()
    num_clades = len(clades_with_color)

    # Define colors for clades
    clade_cmap = ListedColormap(['#d3d3d3', '#ffca8a',  '#99ff8a'])
    colors = plt.get_cmap(clade_cmap, num_clades).colors

    for clade, color in zip(clades_with_color, colors):
        subset = ancestors_df[ancestors_df['Clade'] == clade]
        ax.scatter(subset['umap1'], subset['umap2'], label=clade, color=color, alpha=0.7, s=50)

    # Overlay predictions with new colors
    prediction_df = mutations_df.dropna(subset=['overall_prediction'])
    unique_predictions = prediction_df['overall_prediction'].unique()
    print(f'unique predictions: {unique_predictions}')

    # Define new cmap for predictions
    prediction_cmap = ListedColormap(['mediumorchid', 'red', 'royalblue', 'forestgreen', 'teal'])
    if unique_predictions[0] == 'NR4':
        prediction_cmap = ListedColormap(['teal', 'forestgreen', 'royalblue', 'red', 'mediumorchid'])

    prediction_colors = plt.get_cmap(prediction_cmap, 5).colors

    for prediction, color in zip(unique_predictions, prediction_colors):
        pred_subset = prediction_df[prediction_df['overall_prediction'] == prediction]
        plt.scatter(pred_subset['umap1'], pred_subset['umap2'], 
                    color=color, label=f'Prediction: {prediction}', alpha=0.6, s=50)

    # Set plot titles and labels
    plt.title("UMAP by Clade")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.savefig(outpath)



def main():
    # Load the df with the mutated sequences
    with open(snakemake.input.embedding_df, "rb") as input_file:
        embedding_df = pickle.load(input_file)

    # load the df with the prediction info
    with open(snakemake.input.predictions_df, "rb") as prediction_input:
        prediction_df = pd.read_csv(prediction_input)
    
    embedding_predictions = pd.merge(embedding_df, prediction_df[['info', 'overall_prediction']], 
                                     on='info', how='left')

    nodes_to_label = embedding_df['info'].values
    # print('Nodes to label:', nodes_to_label)

    # Load previously calculated ancestor embeddings
    # with open("./data/ancestor_embedding_df.csv", "rb") as input_file:
    #     ancestor_embedding_df = pickle.load(input_file)
    with open(snakemake.input.ancestor_embeddings, "rb") as input_file:
        ancestor_embedding_df = pickle.load(input_file)

    dataset_name = snakemake.wildcards.dataset_name

    # ancestor_embedding_df['Clade'] = ancestor_embedding_df['info'].apply(seq_utils.tag_node, dataset=dataset_name)
    ancestor_embedding_df['Clade'] = ancestor_embedding_df['info'].apply(seq_utils.tag_node, dataset='combined')

    # # Filter for only NR1 or NR4 clades
    specific_ancestor_embedding_df = ancestor_embedding_df[ancestor_embedding_df['Clade'].isin(['NR1', 'NR4'])]
    # specific_ancestor_embedding_df = ancestor_embedding_df


    # Concatenate the embeddings and ancestor embeddings
    # all_embeddings_df = pd.concat([embedding_df, specific_ancestor_embedding_df])

    # Plot PCA
    # plot_pca(all_embeddings_df, nodes_to_label, snakemake.output.plot_mutation)
    plot_pca_ancestors_static(embedding_df, specific_ancestor_embedding_df, nodes_to_label, snakemake.output.plot_mutation)
    # plot_tsne_ancestors_static(embedding_df, specific_ancestor_embedding_df, nodes_to_label, snakemake.output.plot_mutation)


    # all_embeddings_prediction_df = pd.concat([embedding_predictions, specific_ancestor_embedding_df])
    # mutation_prediction_df = pd.concat([embedding_predictions, embedding_df])
    # plot_pca_colour_by_predicted(all_embeddings_prediction_df, nodes_to_label, snakemake.output.plot_prediction)
    plot_pca_colour_by_predicted_ancestors_static(embedding_predictions, specific_ancestor_embedding_df, nodes_to_label, snakemake.output.plot_prediction)
    # plot_tsne_colour_by_predicted_ancestors_static(embedding_predictions, specific_ancestor_embedding_df, nodes_to_label, snakemake.output.plot_prediction)
    # plot_umap_colour_by_predicted_ancestors_static(embedding_predictions, specific_ancestor_embedding_df, nodes_to_label, snakemake.output.plot_prediction)




if __name__ == "__main__":
    main()
