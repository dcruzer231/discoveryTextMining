from parseDiscoveryRequests import getEmbeds, countRequests
import os
import numpy as np
import pandas as pd
import os
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer


# input_dir = "RFA"

# input_pdf_paths = sorted(
#     [
#         os.path.join(input_dir, fname)
#         for fname in os.listdir(input_dir)
#         if fname.endswith(".pdf")
#     ]
# )

# pdf1 = '.\\discoveries\\BUCHANAN & BUCHANAN, P.A..pdf'
# pdf2 = '.\\discoveries\\Evangelo, Brandt & Lippert, P.A.-2.pdf'
# pdf3 = '.\\discoveries\\ANDREW J. GORMAN & ASSOCIATES.pdf'

def dbscan_cluster(e,r,eps=0.3):
    e_cosine = e / length
    X_embedded = TSNE(n_components=3, metric="cosine",learning_rate='auto',init='random', perplexity=400).fit_transform(e_cosine)
    train_r = r[:761]
    train_X = X_embedded[:761]
    test_X = X_embedded[761:]
    test_r = r[761:]
    db = DBSCAN(eps=eps, min_samples=10, metric="cosine").fit(train_X)
    return db

def silhouetteGraph(x,n_max=600,sil_step=10):
    sil_step = 10
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    # Notice you start at 2 clusters for silhouette coefficient
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    for k in range(2, n_max,sil_step):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(x)
        score = silhouette_score(x, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, n_max,sil_step), silhouette_coefficients)
    plt.xticks(range(2, n_max,sil_step))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
    return silhouette_coefficients

def graphSSE(x,n_max=600,sil_step=10):
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, n_max,sil_step):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(x)
        sse.append(kmeans.inertia_)
    
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, n_max,sil_step), sse)
    plt.xticks(range(1, n_max,sil_step))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    return np.array(sse)

def graphCLusterSizes(kmean):
    ncluster = kmean.cluster_centers_.shape[0]
    labels = kmean.labels_
    y = []
    for i in range(ncluster):
        count = np.sum(labels==i)
        y.append(count)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(np.arange(ncluster),y)
    return np.array(y)
        
def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)

def compute_gap(clustering, data, k_max=5, n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max+1):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))
    
    ondata_inertia = []
    for k in range(1, k_max+1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))
        
    gap = np.log(reference_inertia)-np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)

def plotgap(X):
    k_max = 5
    gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), X, k_max)


    plt.plot(range(1, k_max+1), reference_inertia,
             '-o', label='reference')
    plt.plot(range(1, k_max+1), ondata_inertia,
             '-o', label='data')
    plt.xlabel('k')
    plt.ylabel('log(inertia)')
    plt.show()

def dbscanrequests(filenames):
    r,e = getEmbeds(filenames)
    dbs = DBSCAN(eps=0.05, min_samples=5, metric="cosine")
    dbs.fit(e)
    df = pd.DataFrame({"request":r, "cluster":dbs.labels_})
    return df
    
if __name__ == '__main__':
    pdfs = [pdf1,pdf2,pdf3]
    pdfs = input_pdf_paths
    r,e = getEmbeds(pdfs)


    length = np.sqrt((e**2).sum(axis=1))[:,None]
    e_cosine = e / length

    num_req = len(r)

    random.seed(1773)
    ik = np.array(random.sample(range(num_req), num_req))
    e_r = e[ik]
    r_r = [r[a] for a in ik]

    pca = PCA(20)
    pca_results = pca.fit_transform(e_r)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    dbs = DBSCAN(eps=0.05, min_samples=5, metric="cosine")
    dbs.fit(e_r)

    df = pd.DataFrame({"request":r_r, "cluster":dbs.labels_})
#df.to_csv("dbs_cosdist_lower.csv",encoding="cp1252",index=False)

# train_r = r_r[:761]
# train_e = e_r[:761]
# test_e = e_r[761:]
# test_r = r_r[761:]

# length = np.sqrt((train_e**2).sum(axis=1))[:,None]
# train_e_cosine = train_e / length

# length = np.sqrt((test_e**2).sum(axis=1))[:,None]
# test_e_cosine = test_e / length




# X_embedded = TSNE(n_components=3, metric="cosine",learning_rate='auto',init='random', perplexity=400).fit_transform(train_e_cosine)