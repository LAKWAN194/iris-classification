import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pairplot(df, target_column='class', save_path=None):
    """
    Plot every pair of features against each other, coloured by class

    If save_path is given the figure is saved there instead of being shown
    """
    grid = sns.pairplot(df, hue=target_column, diag_kind='hist', height=2.2)
    grid.figure.suptitle("Iris Features by Class", y=1.02)

    if save_path:
        grid.savefig(save_path, dpi=150)
        plt.close(grid.figure)
        print(f"Pairplot saved to {save_path}")
    else:
        plt.show()

def plot_pca(X, class_labels, figsize=(8, 6), save_path=None):
    """
    Project the four features onto two principal components and plot them

    If save_path is given the figure is saved there instead of being shown
    """
    # Scale first so no single measurement dominates the components
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame({
        'PC1': components[:, 0],
        'PC2': components[:, 1],
        'class': list(class_labels)
    })

    plt.figure(figsize=figsize)
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='class', s=60)
    plt.title('PCA of Iris Features')
    plt.xlabel(f'PC1 ({explained[0]:.1f}% of variance)')
    plt.ylabel(f'PC2 ({explained[1]:.1f}% of variance)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"PCA plot saved to {save_path}")
    else:
        plt.show()

    return pca.explained_variance_ratio_
