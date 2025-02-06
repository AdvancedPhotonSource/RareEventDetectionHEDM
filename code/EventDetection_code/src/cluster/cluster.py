"""

The base Clustering class.


"""

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import joblib

class Cluster:
    def __init__(self, numClusters=30):
        self._numClusters = numClusters
        self._modelName   = "Kmeans"
        self._kmm         = None


    def train(self, emb):
        self._kmm = KMeans(n_clusters=self._numClusters, random_state=2022).fit(emb)
        joblib.dump(self._kmm, "kmeans_model.pkl")
        return self._kmm


    def soft_clustering_weights(self, data, cluster_centres, **kwargs):

        """
        Function to calculate the weights from soft k-means
        data: Array of data. Features arranged across the columns with each row being a different data point
        cluster_centres: array of cluster centres. Input kmeans.cluster_centres_ directly.
        param: m - keyword argument, fuzziness of the clustering. Default 2
        """

        # Fuzziness parameter m>=1. Where m=1 => hard segmentation
        m = 2
        if 'm' in kwargs:
            m = kwargs['m']

        Nclusters = cluster_centres.shape[0]
        Ndp = data.shape[0]
        Nfeatures = data.shape[1]

        # Get distances from the cluster centres for each data point and each cluster
        EuclidDist = np.zeros((Ndp, Nclusters))
        for i in range(Nclusters):
            EuclidDist[:,i] = np.sum((data-np.matlib.repmat(cluster_centres[i], Ndp, 1))**2,axis=1)

        # Denominator of the weight from wikipedia:
        invWeight = EuclidDist**(2/(m-1))*np.matlib.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
        Weight = 1./invWeight

        return Weight

    def kmeans_clustering_and_dist(self, emb, min_score=0.1):
        self._kmm = joblib.load("kmeans_model.pkl")
        pred   = self._kmm.predict(emb)
        counts = np.array([(pred==_c).sum() for _c in range(self._kmm.n_clusters)])
        dist   = counts / counts.sum()
    
        weights = self.soft_clustering_weights(emb, self._kmm.cluster_centers_)
        confidence = np.max(weights, axis=1)   
        uq_score = (confidence < min_score).sum() / confidence.shape[0]

        return uq_score, dist

