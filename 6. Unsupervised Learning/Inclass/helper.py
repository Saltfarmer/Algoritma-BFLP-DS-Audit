import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px

# plt.style.use('seaborn')


def biplot_pca(data):
    """
    Function to perform plots with first 2 PC from PCA on selected data
    Data needs to be scales for the better result
    
    data = selected dataframe
    """
    
    pca = PCA()
    pca.fit(data.values)
    eig_vec_pc1 = pca.components_[0]
    eig_vec_pc2 = pca.components_[1]
    value_pc1 = pca.transform(data)[:,0]
    value_pc2 = pca.transform(data)[:,1]
    for i in range(len(eig_vec_pc1)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
        plt.arrow(0, 0, eig_vec_pc1[i]*max(value_pc1), eig_vec_pc2[i]*max(value_pc2),
                  color='yellow', width=0.0005, head_width=0.0025)
        plt.text(eig_vec_pc1[i]*max(value_pc1)*1.2, eig_vec_pc2[i]*max(value_pc2)*1.2,
                 list(data.columns.values)[i], color='magenta')

    for i in range(len(value_pc1)):
    # circles project documents (ie rows from csv) as points onto PC axes
        plt.scatter(value_pc1[i], value_pc2[i], c='grey')
        plt.text(value_pc1[i]*1.2, value_pc2[i]*1.2, list(data.index)[i], color='brown')
    plt.title('Biplot PCA', fontsize=20)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    
    return plt.show()


def biplot_plotly(data, obj_pca):
    # ambil nama tiap kolom balance_saled sesuai urutan di df
    features = list(data[0:100].columns.values) 
    # hasil PCA. sama seperti objek transform_ di atas, tapi beda nama kolom
    components = obj_pca.fit_transform(data[0:100]) 
    # dot product/perkalian matriks antara eigenvector dengan akar kuadrat eigen value
    loadings = obj_pca.components_.T * np.sqrt(obj_pca.explained_variance_) 

    # Pilih fitur dengan loadings terbesar pada PC1 dan PC2
    important_features = [features[i][0] for i in np.abs(loadings[:, :5]).argmax(axis=0)]

    fig = px.scatter(components, x=0, y=1, hover_name=list(range(len(components))))

    for i, feature in enumerate(features):
        # Hanya tambahkan garis panah dan anotasi untuk fitur yang penting
        if feature[0] in important_features:
            fig.add_shape(
                type='line',
                x0=0, y0=0,
                x1=loadings[i, 0],
                y1=loadings[i, 1],
            )

            fig.add_annotation(
                x=loadings[i, 0],
                y=loadings[i, 1],
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=features[i][0], 
            )

            fig.add_annotation(
                x=loadings[i, 0],
                y=loadings[i, 1],
                ax=0, ay=0,
                xref='x', yref='y',
                axref='x', ayref='y',
                text='',  # if you want only the arrow
                showarrow=True,
                arrowhead=4,
                arrowsize=1.5,
                arrowwidth=1,
                arrowcolor='black'
            )

    # Plotly figure layout
    fig.update_layout(title='Biplot PCA', title_x=0.5, width=800, height=600)
    fig.update_xaxes(title_text='PC1')
    fig.update_yaxes(title_text='PC2')
    fig.show()