#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Learning
# 
# Supervised learning is dedicated to finding conditional relationships $Y_i|\boldsymbol{X}_i$ between feature variables $\boldsymbol{X}_i$ and a target variable $Y_i$. Unsupervised learning is not related to a target variable, but rather is focused on the multivariate distribution of $\boldsymbol{X}_i$. More concrete, without labeling a target variable, we are interested in structure and relationships among the variables $\boldsymbol{X}_i$. The most frequently used tasks with this respect are **clustering** and **dimensionality reduction**. As the name suggests, clustering is dedicated to finding clusters of observations which are more similar to each other than to observations from a different cluster. This type of knowledge can be of great usage, for instance, to segment customers or companies. 
# 
# Dimensionality reduction comes with the need to deal with the **curse of dimensionality**. Especially, if we are interested in many features at the same time, dimensionality increases. For higher dimensions, single observations tend to be less similar to each other and to be farer away in space. If you think about supervised learning problems, the task is to find models which describe the relationship between features and targets well, such that predictions for similar observations can be made and are reliable due to a certain level of similar behavior. The higher dimensionality, the less reliable such assessments can become, because it is less likely to find similar observations.
# 
# Thus, in supervised as well as unsupervised learning, one is interested in keeping dimensionality rather low, without loosing too much information due to dimensionality reduction. This is why techniques for dimensionality reduction are important for unsupervised as well as supervised learning problems. With respect to unsupervised learning, the aim is to find similar low dimensional representations of high dimensional data. If this succeeds, one is able to reveal systematic "latent" patters within the data. For instance, can we reveal a systematic behavior of financial markets which includes as much information as possible for market price movements of single companies? Understanding systematic patters is of great importance and benefit for many different financial tasks. Once, we are able to reduce the dimensionality without the loss of information, we can use lower dimensional feature spaces for supervised learning which may benefit from lower dimensionality as explained before.
# 
# Several approaches for clustering and dimensionality reduction exists. At first, we focus on one method for each task. This will provide an idea how such approaches work and enables readers to better understand and study other approaches on their own.

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# 04_01_kmeans_clustering
# 04_02_pca
# ```
# 
