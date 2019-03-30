import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

movies = ["Die Hard", "Matrix", "Titanic", "Amelie"]
john =  [ 5, 4.5, 3,   2.5]
jack =  [ 4, 5,   3.5, 2]
jessy = [ 3, 3,   3.5, 3.5]
jacob = [ 3, 2.5, 4,   4.5]
diane = [ 2, 3.5, 4,   5]

simple_df = pd.DataFrame(data=[john, jack, jessy, jacob, diane], columns=movies)

#PCA
pca = PCA(n_components=2, svd_solver="full")
reduced = pca.fit_transform(simple_df.values)
#reduced

#TRUNCATED SVD
svd = TruncatedSVD(n_components=2, n_iter=500)
svd_reduced = svd.fit(simple_df.values)
svd_reduced.singular_values_
u, s, vh = np.linalg.svd(simple_df.values, full_matrices=False)

