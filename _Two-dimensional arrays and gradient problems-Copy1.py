#!/usr/bin/env python
# coding: utf-8

# In[2]:


#problem 1

import numpy as np
x_ndarray = np.arange(-50, 50.1, 0.1)
y_ndarray = 0.5*x_ndarray + 1
x_ndarray, y_ndarray


# In[3]:


#problem 2
xy_ndarray = np.stack((x_ndarray, y_ndarray),-1)
print(xy_ndarray)
xy_ndarray.shape


# In[5]:


#problem 3
dx = np.diff(x_ndarray)
dy = np.diff(y_ndarray)
slope = dy/dx
slope = slope.reshape(slope.shape[0],1)

slope.shape


# In[8]:


#problem 4
import matplotlib.pyplot as plt
plt.xlabel("x")
plt.xlabel("gradient")
plt.title("linear funtion")
plt.plot(x_ndarray,y_ndarray, color='orange' ,linewidth=3)
plt.show()


# In[ ]:





# In[19]:


#problem 5
def compute_gradient(funtion, x_range = (-50, 50.1, 0.1)):
    x_array = np.arange(*x_range)
    y_array = funtion(x_array)
    
    xy_array = np.concatenate((x_array[:,np.neaaxis], y_array[:,newaxis]), axis=1)
    gradient = (xy_array[1:, 1] - xy_array[:-1, 1])/(xy_array[1:,0] - xy_array[:L-1, 0])
    def funtion(x_array):
        y_array = x_array**2
        return y_array
    xy_array1, gradient1 = copute_gradient(funtion1)
    print(compute_gradient(funtion1))


# In[20]:


#problem 6
def compute_gradient(funtion, x_range = (-50, 50.1, 0.1)):
    array_x = np.arange(*x_range)
    array_y = funtion(x_array)
    min_y_value =np.min(array_y)
    min__arg =np.argmin(array_y)
    array_xy = np.stack((array_x, array_y),-1)
    return f'The minimum value of y for this funtion is {min_y_value} and its index is {min_y_arg}'


# In[ ]:





# In[ ]:




