#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris_df = sns.load_dataset('iris')
iris_df.head()


# In[2]:


#Observar a quantidade de elementos de cada classe


# In[8]:


iris_df.species.value_counts()


# In[10]:


iris_df.describe()


# In[11]:


sns.pairplot(iris_df, hue='species')


# In[ ]:


#Fit do modelo de árvore de decisão


# In[14]:


X = iris_df.iloc[:, 2:4]
y = iris_df.species


# In[17]:


tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)


# In[18]:


petal_size = 6
petal_width = 1.2

tree_clf.predict([
    [petal_size, petal_width]
])


# In[20]:


pip install pydotplus


# In[1]:


from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

classes = iris_df.species.unique()

export_graphviz(
    tree_clf,
    out_file=dot_data,
    filled=True,
    feature_names=iris_df.columns[2:4],
    class_names= classes
)

graph = pydotplus.graph_from_dot_data(dot_data_getvalue())
Image(graph.create_png())


# In[ ]:
