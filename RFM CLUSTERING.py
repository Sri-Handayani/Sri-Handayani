#!/usr/bin/env python
# coding: utf-8

# # Load libraries

# In[2]:


get_ipython().system('pip install plotly')


# In[3]:


import numpy as np
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
import plotly.express as px

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

from scipy.cluster.hierarchy import dendrogram, linkage


import warnings
import os
warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected = True)


# # Load and Describe Data

# In[29]:


df = pd.read_csv('transactions.csv')


# In[30]:


df.head(5)


# In[31]:


df.info()


# In[32]:


df.shape


# # Check Missing & Null Data

# In[33]:


data_missing_value = df.isnull().sum().reset_index()
data_missing_value.columns = ['feature','missing_value']
data_missing_value


# In[34]:


df.duplicated().sum()


# In[35]:


df=df.drop_duplicates()


# In[36]:


df.duplicated().sum()


# # Recency

# In[39]:


df['trans_date'] = pd.to_datetime(df['trans_date'],errors='coerce')


# In[41]:


df['trans_date'] = df['trans_date'].astype('datetime64[ns]')


# In[42]:


print(df['trans_date'].min(), df['trans_date'].max())


# In[43]:


last_date=df['trans_date'].max()
last_date


# In[44]:


recency=df.groupby('customer_id').agg({'trans_date': 'max'}).reset_index()
recency.columns=['customer_id','last_trans']
recency['recency']=last_date-recency['last_trans']
recency=recency.drop(['last_trans'],axis=1)

recency['recency'] = pd.DataFrame(recency['recency'].astype('timedelta64[D]'))

recency


# # Frequency

# In[46]:


frequency=df.groupby('customer_id').agg({'trans_amount':'count'}).reset_index()
frequency.columns=['customer_id','frequency']
frequency


# # Monetary

# In[48]:


monetary=df.groupby('customer_id').agg({'trans_amount': 'sum'}).reset_index()
monetary.columns=['customer_id','monetary']
monetary


# # RFM

# In[49]:


RFM=recency.merge(frequency,
                 on='customer_id')
RFM=RFM.merge(monetary,
             on='customer_id')
RFM


# # Standarization

# In[50]:


X=RFM[['recency','frequency','monetary']]

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
RFM_std = pd.DataFrame(data = X_std, columns = ['recency','frequency','monetary'])
RFM_std.describe()


# # K Means

# In[56]:


k_max = 10
inertia = []
silhouette = []

for k in range(2, k_max):
    km =  KMeans(init = 'k-means++', n_clusters = k, random_state= 49)
    km.fit(RFM_std.values)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X, km.labels_))

plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(2 , k_max) , inertia , 'o')
plt.plot(np.arange(2 , k_max) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.title('The Elbow method using Inertia for each number of cluster')
plt.show()

plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(2 , k_max) , silhouette , 'o')
plt.plot(np.arange(2 , k_max) , silhouette , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Silhouette')
plt.title('Silhouette score for each number of cluster')
plt.show()


# from the test results above we will use k=4.

# In[57]:


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=49)
kmeans.fit(RFM_std.values)


# In[58]:


RFM_std['cluster']= kmeans.labels_


# In[59]:


melted_RFM=pd.melt(RFM_std.reset_index(),
                  id_vars=['cluster'],
                   value_vars=['recency','monetary','frequency'],
                   var_name='Features',
                   value_name='Value'
                  )

sns.lineplot('Features','Value',hue='cluster',data=melted_RFM)
plt.legend()


# In[60]:


fig = px.scatter_3d(RFM_std, x='recency', y='monetary', z='frequency', color='cluster',
                   opacity = 0.8, height=800)
fig.show() # to show as static image: fig.show('svg') 


# # Clustering Analysis

# Loyal customers have low recency values and high frequency and monetary values, while regular customers have high recency values and lower frequency and monetary values. It can be seen in the recency,frequency, and monetary diagram that the order of customers from most loyal to regular customers is cluster 0, 2, 1, 3.
# 
# Then we label it
# 
#     - High Loyalty (cluster 0)
#     - Medium Loyalty (cluster 2)
#     - Low Loyalty (cluster 1)
#     - No Loyalty (cluster 3)

# In[84]:


RFM['Cluster']=RFM_std['cluster']


# In[85]:


RFM.head()


# In[88]:


RFM.replace([0,1,2,3],['High Loyalty','Low Loyalty','Medium Loyalty','No Loyalty'])


# In[ ]:




