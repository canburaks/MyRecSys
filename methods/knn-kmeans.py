from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd
import numpy as np
import _pickle as pickle
from array import array
import pickson
from tqdm import tqdm
import random
import plotly.offline as pyo
import plotly.graph_objs as go
pd.set_option('display.max_columns', 800)
pd.set_option('display.max_rows', 800)

media_folder = "/home/jb/Projects/Github/RecommenderSystem/media"
save_to = media_folder + "/clustering.html"

#MOVIE-TAG DATAFRAME
mt_full = pd.read_csv("/home/jb/Projects/Github/MyRecSys/movielens/filtered-data/filtered_tags1_score.csv")

# REDUCE DIMENSIONS


def sample(sample_size=None):
    if sample_size == None:
        features = mt_full.iloc[:, 2:].values
    else:
        features = mt_full.iloc[:sample_size, 2:].values
    #return sampled data values
    return StandardScaler().fit_transform(features)


def pca_reduction(features, componens_num, svd_solver="auto"):
    pca = PCA(n_components=componens_num, svd_solver=svd_solver)
    #TRANSFORM
    p_components = pca.fit_transform(features)
    principalDf = pd.DataFrame(data=p_components,
                               columns=["PC" + str(x+1) for x in range(componens_num)])
    finalDf = pd.concat(
        [mt_full["movie_id"], mt_full["name"], principalDf], axis=1)
    return finalDf

##################################################################

def take_sample(df, sample_size=None):
    if sample_size == None:
        return df
    else:
        #create same random int for both dataframes
        sample_ids = np.random.randint(0, df.shape[0]-1, sample_size)
        return df.iloc[sample_ids, :]



def get_color(id):
    return 'hsl({}, 50%, 50%)'.format(id)


def elements2d(movie, label, text=None):
    return go.Scatter(x=[movie[0]], y=[movie[1]],
                      mode="markers",
                      name=str(labels_unique[label]),
                      text=text,
                      marker=dict(size=12,
                                  color=get_color((360/cluster_number)*label),
                                  symbol="circle",
                                  line={"width": 2}))


def elements3d(movie, label, text=None):
    return go.Scatter3d(x=[movie[0]], y=[movie[1]], z=[movie[2]],
                      mode="markers",
                      name=str(labels_unique[label]),
                      text=text,
                      marker=dict(size=12,
                                  color=get_color((360/cluster_number)*label),
                                  symbol="circle",
                                  line={"width": 2}))

def make_graph(sample_size, cluster_number, dim):
    data = sample()
    result = pca_reduction(data, dim)
    #KNN PART - KMEANS
    classifying_data = take_sample(result, 1000).iloc[:, 2:].values
    classified_movie_names = take_sample(result, 1000).iloc[:, 1].values
    #KMEANS
    kmeans = KMeans(n_clusters=cluster_number,random_state=0).fit(classifying_data)
    labels = kmeans.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters: {}".format(n_clusters_))
    if dim==2:
        elements = elements2d
    elif dim==3:
        elements = elements3d
    #ADD MOVIE POINTS
    data_movie = []
    for index in range(classifying_data.shape[0]):
        movie = classifying_data[index]
        label = kmeans.labels_[index]
        text = classified_movie_names[index]
        data_movie.append(elements(movie, label, text))


    layout = go.Layout(title="Basic Clustering Plot",
                    xaxis={"title": "Principal Component 1"},
                    yaxis=dict(title="Principal Component 2"),
                    hovermode="closest")

    data_show = data_movie

    fig = go.Figure(data=data_show, layout=layout)
    pyo.plot(fig, filename=save_to)


make_graph(1000, 10, 3)


"""
data = sample()


result = pca_reduction(data, 2)

#KNN PART - KMEANS
classifying_data = take_sample(result, 1000).iloc[:, 2:].values
classified_movie_names = take_sample(result, 1000).iloc[:, 1].values


cluster_number = 10
kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(classifying_data)
labels = kmeans.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters: {}".format(n_clusters_))




#ADD MOVIE POINTS
data_movie = []
for index in range(classifying_data.shape[0]):
    movie = classifying_data[index]
    label = kmeans.labels_[index]
    text = classified_movie_names[index]
    data_movie.append(elements(movie, label, text))


layout = go.Layout(title="Basic Clustering Plot",
                   xaxis={"title": "Principal Component 1"},
                   yaxis=dict(title="Principal Component 2"),
                   hovermode="closest")

data_show = data_movie

fig = go.Figure(data=data_show, layout=layout)
pyo.plot(fig, filename=save_to)
"""
