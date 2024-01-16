import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import sys
sys.path.append("../")


def func_dimRed(data, random_state=11):
    import umap
    ''' Calculate embeddings '''
    pca = PCA(n_components=2).fit(data)
    embdedding_pca = pd.DataFrame(pca.transform(data), columns=["pc1", "pc2"])
    print(f"PC1 (explained variance: {round(pca.explained_variance_ratio_[0],2)})")
    print(f"PC2 (explained variance: {round(pca.explained_variance_ratio_[1],2)})")

    reducer = umap.UMAP(random_state=random_state) #12
    embedding_UMAP = pd.DataFrame(reducer.fit_transform(data), columns=["umap1", "umap2"])

    ## Takes forever?
    ###embedding_tsne = pd.DataFrame(TSNE(random_state=random_state).fit_transform(pca_umap.transform(data)), columns=["tsne1", "tsne2"]) ##12

    return pd.concat([embdedding_pca, embedding_UMAP], axis=1) #embedding_tsne


def func_dimRed_PCAfirst(data, random_state=11):
    import umap
    ''' Calculate embeddings '''
    pca = PCA(n_components=2).fit(data)
    embdedding_pca = pd.DataFrame(pca.transform(data), columns=["pc1", "pc2"])
    print(f"PC1 (explained variance: {round(pca.explained_variance_ratio_[0],2)})")
    print(f"PC2 (explained variance: {round(pca.explained_variance_ratio_[1],2)})")

    print("Doing umap")
    pca_umap = PCA(n_components=5).fit(data)
    reducer = umap.UMAP(random_state=random_state) #12
    embedding_UMAP = pd.DataFrame(reducer.fit_transform(pca_umap.transform(data)), columns=["umap1", "umap2"])
    print("finished")

    ## Takes forever?
    ###embedding_tsne = pd.DataFrame(TSNE(random_state=random_state).fit_transform(pca_umap.transform(data)), columns=["tsne1", "tsne2"]) ##12

    return pd.concat([embdedding_pca, embedding_UMAP], axis=1) #embedding_tsne


def func_plot_dimRed(df_dimRed, df_inverseTransform, varInterest=False, savefig=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", font_scale=1.2)

    ### Set coloring
    if varInterest:
        coloring = df_inverseTransform[varInterest].copy().reset_index().drop(["pid"], axis=1).iloc[:,0]
    else: 
        coloring = pd.Series(np.zeros(df_inverseTransform.shape[0])).astype("int64")#.reset_index() 
    if coloring.dtype == "float64":
        cmap = "inferno"
    else:
        cmap = "coolwarm"  #"Dark2"

    df_dimRed = pd.concat([df_dimRed, coloring], axis=1, ignore_index=True)  
    df_dimRed.columns = ["pc1", "pc2", "umap1", "umap2", "coloring"]
    #print(df_dimRed)

    fig, (ax1,ax3) = plt.subplots(1,2,figsize=(13,6))
    ax1.scatter(df_dimRed["pc1"], df_dimRed["pc2"], c=df_dimRed.iloc[:,-1], s=20, cmap=cmap)
    #ax2.scatter(df_dimRed["tsne1"], df_dimRed["tsne2"], c=df_dimRed.iloc[:,-1], s=20, cmap=cmap)
    plot = ax3.scatter(df_dimRed["umap1"], df_dimRed["umap2"], c=df_dimRed.iloc[:,-1], s=20, cmap=cmap)

    ### Stylise labels
    ax1.set_xlabel(f"PC1")
    ax1.set_ylabel(f"PC2")
    ax1.set_title("PCA")
    # ax2.set_xlabel(f"TSNE1")
    # ax2.set_ylabel(f"TSNE2")
    # ax2.set_title("TSNE")
    ax3.set_xlabel(f"UMAP1")
    ax3.set_ylabel(f"UMAP2")
    ax3.set_title("UMAP")

    #### Add correct labels
    if coloring.dtype == "float64":
        cbar = plt.colorbar(plot)
        cbar.ax.set_ylabel(varInterest)
    else:
        legend = ax3.legend(*plot.legend_elements(),loc="lower left", title=varInterest)
        ax3.add_artist(legend)
        sns.move_legend(ax3, "upper left", bbox_to_anchor=(1, 1))

    if savefig:
        os.makedirs(f"figures/clustering", exist_ok=True)
        plt.tight_layout()
        fig.savefig(f"figures/clustering/clustering_{varInterest}.png", dpi=300)    
    return 
