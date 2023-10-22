#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np


# In[49]:


df = pd.read_csv("E:\Missing_Data.csv")   # seperated by ;
print(df.head(2))


# In[50]:


print(df.shape) # Data Validation


# In[51]:


print(df.info())


# In[52]:


data = df.copy()


# In[53]:


df== '?'


# In[54]:


df[df== '?']


# In[55]:


# Filtering     # we replaed the ? as nan
df[df== '?'] = np.nan


# In[13]:


df.head(4)


# In[56]:


df.info()


# In[15]:


df.isnull()


# In[57]:


df.shape[0]


# In[58]:


df.isnull().sum()


# In[18]:


per = round((df.isnull().sum()/df.shape[0])*100,2)
print("the percentage -- \n",per)


# In[59]:


per.head(5)


# In[61]:


type(per)


# In[21]:


pd.DataFrame(per, columns = ['Missing_percent']).sort_values('Missing_percent', ascending = False).head(5)


# In[62]:


type(per)


# In[63]:


df.columns


# In[37]:


#df.drop(['OTHERSTATUS','OTHER_POSITION'],axis = 1,inplace = True) # above 25% of missing columns dropped


# In[64]:


df.isnull().sum()


# In[25]:


df.columns


# In[65]:


df.tail(30)


# In[27]:


df.dtypes


# In[67]:


df.shape[1]  # data with dropped columns


# In[68]:


data.shape[1]  # Actual Data number of column


# In[70]:


df.columns


# In[73]:


# creating subset of column
exp = df[['YEARSEXP','Exp1','Exp2','Exp3','Exp4','Exp5']]
print(exp.shape)
print(exp.dtypes)   # we need to convert this column to int datatypes
print(exp.isnull().sum()) # we need to handle missing value 


# In[74]:


exp.head(3)


# In[75]:


exp['YEARSEXP'].astype('float')


# In[78]:


exp1 = exp.astype('float')
print(exp1.dtypes)


# In[80]:


exp.apply(pd.to_numeric)


# In[81]:


print(exp1.isna().sum()) # Missing value imputation


# In[82]:


#replacing missing value by 0
exp1.fillna(0)


# In[86]:


exp1.fillna(exp1.mean())


# In[84]:


#replacing missing value by median
exp1.fillna(exp1.median())


# In[87]:


#replacing missing value by front value
exp1.fillna(method = 'ffill')


# In[89]:


#replacing missing value by front value
exp1.fillna(method = 'bfill')


# In[90]:


###### Seaborn  - Its a very creative package for charts and graphs
##### You can styles your Visualisations very beatifully over here
import seaborn as sns


# In[91]:


sns.get_dataset_names()   ###### Inbuilt Datasets present in Seaborn


# In[92]:


df = sns.load_dataset('tips')
print(df.head(2))


# In[93]:


df.sex.value_counts().plot(kind = 'barh')   #### Pandas 


# In[94]:


df.groupby('sex')['total_bill'].sum().plot(kind = 'bar')


# In[95]:


sns.countplot(x = 'sex', data = df)   #### Seabron


# In[96]:


sns.countplot(x = 'day', data = df,palette= 'spring')


# In[97]:


sns.barplot(x = 'sex', y = 'total_bill', hue ='day', 
            data = df, estimator='sum',errorbar=None)


# In[98]:


sns.barplot(x = 'total_bill', y = 'day',hue = 'sex',
            data = df, estimator='sum',errorbar=None) ### Horizontal Bar chart with distribution of sex


# In[99]:


sns.boxplot(x = df['total_bill'])  #### Total_bill has an outlier


# In[100]:


Q1 = df['total_bill'].quantile(0.25)
Q1


# In[101]:


Q3 = df['total_bill'].quantile(0.75)
Q3


# In[102]:


IQR = Q3 - Q1
IQR


# In[103]:


upper = Q3 + (1.5*IQR)
upper


# In[104]:


lower = Q1 - (1.5*IQR)
lower


# In[105]:


df[df['total_bill']==40.29]


# In[106]:


##### np.where
df['total_bill'] = np.where(df['total_bill']>40.29,40.29,df['total_bill'])


# In[107]:


###### Outliers are Imputed with the Upper Threshold Limit
sns.boxplot(x = df['total_bill'])


# In[108]:


sns.boxplot(x = df['tip'])


# In[109]:


sns.displot(df['tip'])


# In[110]:


df['tip'].skew()


# In[111]:


sns.displot(df['total_bill'])


# In[112]:


c = df[['total_bill','tip','size']].corr()


# In[113]:


sns.heatmap(c,annot = True,cmap = 'RdBu')


# In[ ]:




