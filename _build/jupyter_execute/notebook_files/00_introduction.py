#!/usr/bin/env python
# coding: utf-8

# # What is financial data analytics
# 
# Due to increasing data availability and higher computational power, data analysis has become more and more important to academia and the industry. The financiel area is characterized by a large variety of different data sources which makes it a natural candidate for data driven analysis. 
# 
# As a starting point, we try to create our own definition of <b>financial data analytics</b>: <i>a systematic computational activity to analyze financial data with the aim to gather relevant information in data in order to support business decisions for creating economic value.</i> Analyzing data comes along with many different dimensions. In this book we try to go through an evolutionary process of the most important skills you may find useful for financial data analytics. In particular, this is:
# 
# <ul>
#     <li>Software and programing language</li>
#     <li>Fundamentals of math and statistics</li>
#     <li>Data collection, visualization and preprocessing</li>
#     <li>Data modeling - a supervised generalized approach</li>
#     <li>Validation and overfitting</li>
#     <li>Finding structure in data - unsupervised learning</li>
# </ul>
# 
# Software and programing skills are your most valuable helpers in data analysis and, thus, should be learned right from the start. In my opinion, the best method is to gather a minimum of theoretical knowledge and follow a hands-on approach. This means, you will learn to use software and programing skills in more depth throughout the course when conducting analysis with real data. Like it or not, but math and statistics are the backbone of data analysis. The good news here is that knowledge of fundamental topics and techniques are enough to understand almost all (and partly very sophisticated) methods in data analysis. The next you need to know is how to get data. Different methods for this task exist. Most common is the acces to commcercial data bases, to gather data via (mostly free) APIs (application programing interfaces) from third parties or to scrape data from online and publicly available sources. Once you are in possession of data, you need to preprocess it and visualize it. These three tasks (data collection, preprocessing and visualization) are very important and often neglected in text books. This is why I dedicate almost a complete course to these topics which you can find here. HIER LINK ZU PYTHON KURS. After we have data, we want to learn about it. Typically, this can be achieved either with a supervised and an unsupervised approach. Supervised means that we formulate an aim for our model, a process called labeling. So we define which is our variable of interest and what we are interested in, e.g., categories of this variable or its level. A great number of supervised and unsupervised methods for data analysis exist and the aim of this course is not to discuss them all. However, the aim is to provide a general understanding of these methods which will be helpful to learn other existing methods when needed. The next logical step when we developed a model to analyze data is to validate it. We aim for models which are able to detect general properties of the data generating processes, i.e. we do want our models to work for all data and not only for a random sample - so they should **generalize**. This can be reviewed with validation techniques. Last but not least, and as in many other textbooks, we learn about unsupervised methods. As you may guess, we do not label the data, so there is no direct aim to learn about a particular variable. More generally, supervised learning typically is conducted from a conditional perspective which means given some predictor variable $\boldsymbol{X}$ we want to find out about the target variable $Y$. Unsupervised learning is a rather unconditional perspective which is interested in the distribution among all variables $\boldsymbol{X}^*$. Unsupervised tasks are often dedicated to find systematic patterns which affect all variables. For instance this knowledge can be used to detect outliers, reduce the dimensionality or cluster data. With this knowledge in hand, you should have a solid understanding of data analysis and a good starting point for exploring more sophisticated approaches. 

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :numbered: 
# 
# 01_preliminiaries
# 02_supervised_learning
# 03_neural_networks
# 04_unsupervised_learning
# ```
# 
