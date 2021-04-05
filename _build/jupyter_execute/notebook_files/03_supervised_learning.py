#!/usr/bin/env python
# coding: utf-8

# # Supervised Learning
# 
# In this chapter we will focus on **supervised learning** and how it can be used to analyze financial data. Supervised learning describes a process which consists of a **target** variable $Y_i$ which should be modeled given a set of other variables, usually called **features**, $\boldsymbol{X} = (X_1, ..., X_n)$. The model itself typically consists of a learning algorithm and parameters that need to be estimated from data. The aim is to find a model which generalizes. This means, even though we fit the parameters only to random samples of the data generating process, we hope to find a model that also describes new data in the future as well as the data upon which we estimated the model's parameters. All variables, the target as well as the features can be of categorical or on a metric scale. The type of supervised learning problem is associated with the scale of the target. We distinguish between:
# 
# * Regression problems: $Y_i$ is on a metric scale
# * Classification problems: $Y_i$ is binary
# * Multi-classification problems: $Y_i$ has multiple categories
# 
# Especially in data science, a great focus lies on finding models which outperform others with respect to the prediction task. This is also important for financial applications. However, a great focus also lies on the importance of feature variables. Behind this interest lies the aim to identify important drivers for the target variable. For instance, we may want to predict the revenue which follows certain actions by the management, but we also want to know which part of the actions is most important for the revenue. This is an essential aspect for decision making in economical applications. Depending on the model, the identification of important drivers can get difficult. We will try to set emphasis on identifying so called **feature importance** , which we will discuss for all models in the following sections.
# 
# This chapter will be structure as follows. First, we will take basic modeling approaches for the supervised learning problems from above, namely, **linear regression**, **logistic regression** and **multinomial regression**. We will not dive deep in the theoretic foundation of these models, but rather use them to gain a general understanding how statistical models work. Before we continue with the illustration of other methods, we shortly take a look at performance metrics of statistical models for the three learning problems. Based on this knowledge, we will talk about best practices and important practical aspects such as cross-validation and regularization. We then take a look at further models such as regression/decision trees, support vector machines and neural networks.    
# 
# ...
# 
# 
# 

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# 03_01_baseline_algorithms
# 03_02_blueprint_supervised_learning
# 03_03_tree_models
# ```
# 
