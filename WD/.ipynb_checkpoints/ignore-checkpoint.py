#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


a = np.array([[1,2,3],[4,5,6]])


# In[4]:


len(a)


# In[4]:


import torch
import numpy as np


# In[5]:


field_dims = (6040, 3090)


# In[56]:


offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype = np.long)
offset


# In[57]:


a = torch.tensor([(1, 2), (3,4)])
b = torch.tensor([(1, 2)])


# In[58]:


x2 = x.new_tensor(offset).unsqueeze(0)
x2


# In[60]:


embedding = torch.nn.Embedding(1000000, 16)


# In[52]:


embedding(x2).size()


# In[41]:


torch.sum(b, dim = 1)


# In[62]:


x3 = torch.tensor([3,1])
print(x3.size())
x3 = x3 + x3.new_tensor(offset).unsqueeze(0)
print(x3)
print(x3.size())
embedding(x3).size()


# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


x + x2


# In[15]:


print(torch.zeros((1)))


# In[17]:


fc = torch.nn.Embedding(sum(field_dims), 1)


# In[29]:


fc(x2).size()


# In[25]:


bias = torch.nn.Parameter(torch.zeros((1,)))


# In[26]:


torch.sum(fc(x2), dim = 1) + bias


# In[ ]:




