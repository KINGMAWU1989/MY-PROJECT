#!/usr/bin/env python
# coding: utf-8

# In[4]:


#problem 1
import pandas as pd
pd_train = pd.read_csv('train.csv')
target_variable = pd_train[["GrLivArea","YearBuilt", "SalePrice"]]
target_variable.head()


# In[ ]:





# In[25]:


#problem 2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

x = target_variable.loc[:, ["GrlivArea", "YearBuilt"]]
y = target_variable["salePrice"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_trans = scaler.transform(x_train)
x_test_trans = scaler.transform(x_test)


reg = LinearRegression().fit(x_train, y_train)
reg_pred = reg.predict(x_test_trans)
from sklearn.matrics import mean_squared_error
print(" MSE:",mean_square_error(y_true=y_test, y_pred= reg_pred))


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(x_test.loc[:, 'GrlivArea'], y_test)
plt.title('GrlivArea')
plt.xlabel('GrlivArea')
plt.ylabel('salePrice')
ply.show()


# In[ ]:


#problem 3
mSE_results = []
model_names = []
def display_outputs(title,y_prediction):
    mSE = mean_square_error(y_true=y_test, y_pred=y_prediction)
   
    mSE_results.append(mSE)
    model_names.append(title)
    
    print(title)
    print("MSE:", mSE)
    
plt.scatter(x_test.loc[:, 'GrlivArea'], y_prediction)
plt.title('GrlivArea')
plt.xlabel('GrlivArea')
plt.ylabel('salePrice')
ply.show()

plt.scatter(x_test.loc[:, 'GrlivArea'], y_prediction)
plt.title('GrlivArea')
plt.xlabel('GrlivArea')
plt.ylabel('salePrice')
ply.show()

print("---------------------------------------")


from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

reg = LinearRegression().fit(x_train_trans, y_train)

reg_pred = reg.predict(x_test_trans)


# In[10]:


x = train_df[['GrlivArea', 'YearBuilt']].values
y = train_df['salePrice'].values

x_train, x_test, y_train, y_test = train_test_split(x,y)
print(x_train.shape, x_test.shape, y_test.shape)

reg = LinearRegression().fit(x_train, y_train)

reg_pred = reg.predict(x_test_trans)

displat_outputs('LinearRegression', reg_pred)

clf = 
             


# In[11]:


reg = LinearRegression().fit(x_train, y_train)
reg_pred = reg.predict(x_test_trans)


# In[14]:


print("mean square Error:", mean_squared_error(y_test, reg_predict))


# In[ ]:





# In[15]:


def plot_graph(xlabel, ylabel, x, y_true, y_pred):
    plt.title('visualization between' + xlabel + 'and + ylabel')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)  
    plt.scatter(x, y_true, label='true value')
    plt.scatter(x, y_pred, label='predicted value')
    plt.legend()
    plt.show()


# In[ ]:


plot_graph('GrLiveArea', 'saleprice',x_test[:,0], y_test, reg_predict)


# In[19]:


#problem 4

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[20]:


svr = SVR()
svr.fit(x_train, y_train)
svr_predict = svr.predict(x_test)
print("mean square Error:", mean_squared_error(y_test, reg_predict))


# In[21]:


plot_graph('GrLiveArea', 'saleprice',x_test[:,0], y_test, svr_predict)


# In[ ]:


#problem 4
train_df2 = [["GrLiveArea","YearBuilt", "salesprice""LotArea"," Yrsold"]]
train_df2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




