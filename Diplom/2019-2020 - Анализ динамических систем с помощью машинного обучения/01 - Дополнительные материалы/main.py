import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import NearestNeighbors
import KS_by_CNAB2 as solver
import utils
from seaborn import heatmap

def heatmap_plot(data):
    ax = heatmap(data, xticklabels=False)
    ax.invert_yaxis()
    ax.set_xlabel('x')
    ax.set_ylabel('time')
    plt.show()
    
def find_eps(df):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()
    
def plot_clust(df, clusters):
    df_new = df.drop(['value'], axis='columns')
    df_new['label'] = clusters
    fig, ax = plt.subplots()
    ax.margins(0.03)
    for i in np.unique(clusters):
        temp_df_new = df_new.loc[df_new['label'] == i]
        ax.plot(temp_df_new['x'], temp_df_new['time'], marker='o', linestyle='', ms=1, label=i)
    plt.show()

U,x,t = solver.CNAB2(512)
heatmap_plot(pd.DataFrame(data=U, index=t, columns=x))

xx, tt = np.meshgrid(x, t)
df = pd.DataFrame({'time': tt.ravel(), 'x': xx.ravel(), 'value': U.ravel()})

print(df)
#print("статистика хопкинса = ", utils.hopkins(df))

find_eps(df)
clusters = utils.dbscan_model(df, eps=0.32, samples=3)

plot_clust(df, clusters)
df_new = df.drop(['value'], axis='columns')
df_new['label'] = clusters

fig, ax = plt.subplots()
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling
temp_df = df_new.loc[df_new['label'] == -1]
ax.plot(temp_df['x'], temp_df['time'], marker='o', linestyle='', ms=1, label=-1)
ax.legend()
plt.show()

df['label'] = clusters
temp_df = df.loc[df['label'] == -1]
temp_df = temp_df.drop(['label'], axis='columns')
print(temp_df.shape)

#print("статистика хопкинса = ", hopkins(temp_df))

find_eps(temp_df)
clusters = utils.dbscan_model(temp_df, eps=0.6, samples=4)
plot_clust(temp_df, clusters)

clusters = utils.EM_model(temp_df, n=30, dis_type='spherical')
plot_clust(temp_df, clusters)

clusters = utils.Aglo_model(temp_df, n=30, linkage='average')
plot_clust(temp_df, clusters)