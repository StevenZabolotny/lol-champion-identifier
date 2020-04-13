#!/usr/bin/env python
# coding: utf-8

# In[3]:


from utils import *
from fastai2.vision.widgets import *
from pathlib import Path


# # League of Legends Champion Identifier #
# The goal of this is to use deep learning to create an image classifier which can identify a League of Legends champion by using an image. The image classifier will be trained using 100 images of each of the currently existing League of Legends champions. After creating the classifier, we can test it by giving it images of the champion in different skins to see if the image classifier can actually use characteristics of the champion that it identified in the base 100 images (which may include some but not all of the skins of a champion) to identify the same champion in a different skin

# ## Gathering Data ##

# In[2]:


bing_api_key = 'abc77f12e0fb42e1b15240f49dd9d8af'


# ### Obtaining the Current List of Champions ###
# Navigate to https://na.leagueoflegends.com/en-us/champions/ and use BeautifulSoup to scrape the page. This will allow use to extract the current list of champion names.

# In[3]:


import requests 
from bs4 import BeautifulSoup


# In[4]:


champions_url = "https://na.leagueoflegends.com/en-us/champions/"
champions_response = requests.get(champions_url)
champions_soup = BeautifulSoup(champions_response.content, "html.parser")
print(champions_soup.prettify)


# Go within the structure of the page to extract the champion names. This will require searching for specific divs and spans, and this might need to be updated if the League of Legends champion list page is updated. 
# We are going to find the container that stores all of the champions, get each of the champion containers from within it, and then extract the text from each champion container.

# In[5]:


champions_container = champions_soup.find("div", attrs = {"class": "style__List-ntddd-2"})
champion_containers = champions_container.findChildren("a", recursive = False)
champions = list(map(lambda container: container.findChildren("span", attrs = {"class": "style__Text-sc-12h96bu-3"})[0].text, champion_containers))
print(champions)


# ### Gathering Images ###
# We will use the Bing Image Search API to gather 100 images of each of the League of Legends champions.

# In[5]:


path = Path("league_of_legends_champions")
if not path.exists():
    path.mkdir()
    for champion in champions:
        dest = (path/champion)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(bing_api_key, f'{champion} League of Legends')
        download_images(dest, urls=results.attrgot('content_url'))


# Remove any failed images from the path. 

# In[12]:


verify_images(get_image_files(path)).map(Path.unlink)


# ## Creating and Training the Model ##
# We will now use our downloaded data to create and train an image classifier model using deep learning.

# ### Creating DataLoader ###
# We will use the data and DataBlocks to create an image DataLoader for our League of Legends champion images. In order to accomplish this, we will use a RandomSplitter to create validation data, RandomResizedCrop for our image transformations, and image augmentation for our batch transformations. The label of the image will be extracted from the name of its parent directory.

# In[6]:


champions_db = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.3, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = champions_db.dataloaders(path)
dls.train.show_batch(max_n=4)
dls.valid.show_batch(max_n=4)


# ### Training the Model ###
# We will train the model using the DataLoader, and then analyze and clean the data.

# In[7]:


learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)


# In[15]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[16]:


interp.plot_top_losses(20)


# Cleaning the data doesn't seem to work - possibly because there is too much data. It definitely seems like the data is in need of cleaning though, since based on the top losses, there are definitely some poorly mapped images where the model was actually correct but was marked incorrect. 

# In[ ]:


#cleaner = ImageClassifierCleaner(learn)
#cleaner


# In[8]:


learn.export()


# In[ ]:




