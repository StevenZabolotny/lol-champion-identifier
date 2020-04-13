#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastai2')
get_ipython().system('pip install nbdev')
get_ipython().system('pip install graphviz')
get_ipython().system('pip install azure.cognitiveservices.search.imagesearch')


# In[1]:


from utils import *
from fastai2.vision.widgets import *
from pathlib import Path


# In[2]:


#hide
# !pip install voila
# !jupyter serverextension enable voila --sys-prefix


# # League of Legends Champion Identifier #
# This is a League of Legends Champion Identifier which uses an image classifier trained using deep learning neural network to identify League of Legends champions based on images. The image classifier was trained with 150 images of each champion obtained from the Bing Search API.
# 
# Simply upload an image of a League of Legends champion that you wish to classify and click Classify to see if the identifier can determine the champion in the image. You will also see the probability of that prediction, and the probability of the top 10 other predictions.

# In[3]:


path = Path()


# In[4]:


learn_inf = load_learner(path/'export.pkl')
vocab = learn_inf.dls.vocab


# In[5]:


btn_upload = widgets.FileUpload()


# In[6]:


out_pl = widgets.Output()
out_pl.clear_output()


# In[7]:


lbl_pred = widgets.Label()


# In[8]:


btn_run = widgets.Button(description='Classify')


# In[9]:


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Highest Probability Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[10]:


#hide
#Putting back btn_upload to a widget for next cell
btn_upload = widgets.FileUpload()


# In[11]:


VBox([widgets.Label('Select your champion!'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# In[ ]:




