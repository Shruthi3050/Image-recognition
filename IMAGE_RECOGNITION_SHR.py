#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES

# In[1]:


import os
import warnings
warnings.simplefilter("ignore")


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[5]:


zayn=os.listdir("C:/Users/aishr/Pictures/zayn")
gigi=os.listdir("C:/Users/aishr/Pictures/gigi")
voldemort=os.listdir("C:/Users/aishr/Pictures/voldemort")


# In[6]:


limit=10
zayn_images=[None]*limit
j=0
for i in zayn:
    if(j<limit):
        zayn_images[j]=imread("C:/Users/aishr/Pictures/zayn/"+i)
        j+=1
    else:
        break
        
imshow(zayn_images[4])


# In[7]:


limit=10
gigi_images=[None]*limit
j=0
for i in gigi:
    if(j<limit):
        gigi_images[j]=imread("C:/Users/aishr/Pictures/gigi/"+i)
        j+=1
    else:
        break
        
imshow(gigi_images[4])


# In[8]:


limit=10
voldemort_images=[None]*limit
j=0
for i in voldemort:
    if(j<limit):
        voldemort_images[j]=imread("C:/Users/aishr/Pictures/voldemort/"+i)
        j+=1
    else:
        break
        
imshow(voldemort_images[1])


# In[9]:


zayn_gray=[None]*limit
j=0
for i in zayn:
    if(j<limit):
        zayn_gray[j]=rgb2gray(zayn_images[j])
        j+=1
    else:
        break
        
gigi_gray=[None]*limit
j=0
for i in gigi:
    if(j<limit):
        gigi_gray[j]=rgb2gray(gigi_images[j])
        j+=1
    else:
        break
        
voldemort_gray=[None]*limit
j=0
for i in voldemort:
    if(j<limit):
        voldemort_gray[j]=rgb2gray(voldemort_images[j])
        j+=1
    else:
        break


# In[10]:


imshow(zayn_gray[4])


# In[11]:


imshow(gigi_gray[7])


# In[12]:


imshow(voldemort_gray[1])


# ## RESIZING

# In[13]:


zayn_gray[4].shape

gigi_gray[7].shape

voldemort_gray[1].shape

for j in range(10):
  za=zayn_gray[j]
  zayn_gray[j]=resize(za,(512,512))
    
for j in range(10):
  gi=gigi_gray[j]
  gigi_gray[j]=resize(gi,(512,512))
    
for j in range(10):
  vo=voldemort_gray[j]
  voldemort_gray[j]=resize(vo,(512,512))


# In[14]:


imshow(zayn_gray[4])


# In[15]:


imshow(gigi_gray[7])


# In[16]:


imshow(voldemort_gray[1])


# ## MATRIX TO VECTOR CONVERTION

# In[17]:


len_of_images_zayn=len(zayn_gray)
len_of_images_zayn


# In[18]:


len_of_images_gigi=len(gigi_gray)
len_of_images_gigi


# In[19]:


len_of_images_voldemort=len(voldemort_gray)
len_of_images_voldemort


# In[20]:


image_size_zayn=zayn_gray[1].shape
image_size_zayn


# In[21]:


image_size_gigi=gigi_gray[1].shape
image_size_gigi


# In[22]:


image_size_voldemort=voldemort_gray[1].shape
image_size_voldemort


# ## FLATTENING

# In[23]:


flatten_size_zayn=image_size_zayn[0]*image_size_zayn[1]
flatten_size_zayn


# In[24]:


flatten_size_gigi=image_size_gigi[0]*image_size_gigi[1]
flatten_size_gigi


# In[25]:


flatten_size_voldemort=image_size_voldemort[0]*image_size_voldemort[1]
flatten_size_voldemort


# In[26]:


for i in range(len_of_images_zayn):
  zayn_gray[i]=np.ndarray.flatten(zayn_gray[i]).reshape(flatten_size_zayn,1)
zayn_gray=np.dstack(zayn_gray)
zayn_gray.shape


# In[27]:


for i in range(len_of_images_gigi):
  gigi_gray[i]=np.ndarray.flatten(gigi_gray[i]).reshape(flatten_size_gigi,1)
gigi_gray=np.dstack(gigi_gray)
gigi_gray.shape


# In[28]:


voldemort_gray=np.dstack(voldemort_gray)
voldemort_gray.shape


# In[29]:


zayn_gray=np.rollaxis(zayn_gray,axis=2,start=0)
zayn_gray.shape


# In[30]:


gigi_gray=np.rollaxis(gigi_gray,axis=2,start=0)
gigi_gray.shape


# In[31]:


voldemort_gray=np.rollaxis(voldemort_gray,axis=2,start=0)
voldemort_gray.shape


# In[32]:


zayn_gray=zayn_gray.reshape(len_of_images_zayn,flatten_size_zayn)
zayn_gray.shape


# In[33]:


gigi_gray=gigi_gray.reshape(len_of_images_gigi,flatten_size_gigi)
gigi_gray.shape


# In[34]:


voldemort_gray=voldemort_gray.reshape(len_of_images_voldemort,flatten_size_voldemort)
voldemort_gray.shape


# ## DATA FRAME OF IMAGE VECTOR

# In[35]:


zayn_data=pd.DataFrame(zayn_gray)
zayn_gray


# In[36]:


gigi_data=pd.DataFrame(gigi_gray)
gigi_gray


# In[37]:


voldemort_data=pd.DataFrame(voldemort_gray)
voldemort_gray


# In[38]:


zayn_data["label"]="zayn"
zayn_data


# In[39]:


gigi_data["label"]="gigi"
gigi_data


# In[40]:


voldemort_data["label"]="voldemort"
voldemort_data


# ## CONCAT

# In[41]:


a=pd.concat([zayn_data,voldemort_data])


# In[42]:


final=pd.concat([a,gigi_data])


# In[43]:


final


# In[44]:


from sklearn.utils import shuffle
final_indexed=shuffle(final).reset_index()
final_indexed


# In[45]:


final_list=final_indexed.drop(["index"],axis=1)
final_list


# In[46]:


x=final_list.values[:,:-1]


# In[47]:


y=final_list.values[:,-1]


# In[48]:


x


# In[49]:


y


# ## MODULE SELECTION

# In[50]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# ## SUPPORT VECTOR MACHINE

# In[51]:


from sklearn import svm


# In[52]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# In[53]:


y_pred=clf.predict(x_test)


# In[54]:


y_pred


# In[55]:


for i in (np.random.randint(0,6,4)):
  predicted_images=(np.reshape(x_test[i],(512,512)).astype(np.float64))
  plt.title("Predicted label: {0}".format(y_pred[i]))
  plt.imshow(predicted_images,interpolation="nearest",cmap="gray")
  plt.show()


# ## PREDICTION ACCURACY

# In[56]:


from sklearn import metrics


# In[57]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[58]:


accuracy


# ## ERROR ANALYSIS

# In[59]:


from sklearn.metrics import confusion_matrix


# In[60]:


confusion_matrix(y_test,y_pred)


# In[ ]:




