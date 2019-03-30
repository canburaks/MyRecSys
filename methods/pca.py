from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import _pickle as pickle
from array import array
import pickson
from tqdm import tqdm
import random

pd.set_option('display.max_columns', 800)
pd.set_option('display.max_rows', 800)

#MOVIE-TAG DATAFRAME
mtf_path = "/home/jb/Projects/Github/MyRecSys/movielens/filtered-data/filtered_tags1_score.csv"
mt_full = pd.read_csv(mtf_path)
mt_full
def sample(sample_size=None):
    if sample_size==None:
        features = mt_full.iloc[:, 2:].values
    else:
        features = mt_full.iloc[:sample_size, 2:].values
    #return sampled data values
    return StandardScaler().fit_transform(features)
data = sample()


def pca_reduction(features, componens_num, svd_solver="auto"):
    pca = PCA(n_components=componens_num, svd_solver=svd_solver)
    #TRANSFORM
    p_components = pca.fit_transform(features)
    principalDf = pd.DataFrame(data=p_components, columns=["PC" + str(x+1) for x in range(componens_num)])
    finalDf = pd.concat([mt_full["movie_id"], mt_full["name"], principalDf], axis=1)
    return finalDf

result = pca_reduction(data,2)



#SAVE AS EXCEL FILE
result.to_excel("/home/jb/Projects/Github/MyRecSys/data/principal-components/pc2_df.xlsx", sheet_name="PC2", index=False)


#GET SAMPLE OF 2D AND 3D
r3d = pca_reduction(data, 3)
r2d = pca_reduction(data, 2)
random_indexes = np.random.randint(0, r3d.shape[0], 1000)


sampled_result_3d = r3d.iloc[random_indexes, :]
sampled_result_2d = r2d.iloc[random_indexes, :]

sampled_result_3d.to_excel("/home/jb/Projects/Github/DataVisualization/data/3dim_sample.xlsx", sheet_name="PC3", index=False)
sampled_result_2d.to_excel("/home/jb/Projects/Github/DataVisualization/data/2dim_sample.xlsx", sheet_name="PC2", index=False)
