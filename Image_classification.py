#!/usr/bin/env python
# coding: utf-8

# <h1><b> Image Classifier </h1></b>

# ---
# 

# The dataset consists of two types of images. The files are named <b>Ankit</b> and <b>Harsit</b> and a sample image is displayed below.

# <h3>Ankit </h3>
# 
# 

# <img>![image.png](attachment:image.png)

# <h3> Harsit </h3>

# <img>![image.png](attachment:image.png)

# ---

# In[37]:


#hide
#!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()


# In[38]:


#hide
from fastbook import *
from fastai.vision.widgets import *


# In[39]:


# defines the name of folder where the data is stored in the current directory

path = Path('images')


# In[50]:


# our folder contains image files

fns = get_image_files(path)
fns


# In[ ]:


# ?get_image_files 


# <img>![image.png](attachment:image.png)

# In[94]:


# ?? get_image_files


# <img> ![image.png](attachment:image.png)

# ---

# <h2> Creating object called DATALOADERS </h2>

# <b> DataLoaders is a class that stores whatever dataloader objects we pass to it and makes them available as train and valid. </b>

# <img>![image.png](attachment:image.png)

# In[51]:


#After preparing data, we need to assemble it in suitable format for training model
#For the aforementioned purpose, we need to create an object in fastai called DataLoaders

harsit_ankit = DataBlock(
 blocks=(ImageBlock, CategoryBlock),
 get_items=get_image_files,  #get_image_files function takes a path, and returns a list of all of the images in that path (recursively, by default)
 splitter=RandomSplitter(valid_pct=0.2, seed=42),
 get_y=parent_label, #telling fastai what function to call to create the labels in our dataset
 item_tfms=Resize(128))


# In[52]:


#need to special actual source of data, path where images can be found

dls = harsit_ankit.dataloaders(path)


# In[53]:


# to look for few items by calling show_batch method on DataLoader

dls.valid.show_batch(max_n=6, nrows=1)


# ---

# <h2> Resizing Images</h2>

# <b> Resize crops the images to fit a square shape of size requested using full width or height, can lead to loss of important features. Alternatively, we can pad images with zeros(black) or squish.stretch them. </b>

# In[54]:


#Squishing images

harsit_ankit = harsit_ankit.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = harsit_ankit.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[19]:


#padding

harsit_ankit = harsit_ankit.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = harsit_ankit.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# ---

# <b>  training the neural network with examples of images in which the objects are in slightly different places and are slightly different sizes helps it to understand the basic concept of what an object is, and how it can be represented in an image. </b>

# <h3> RandomResizedTransform (specific example of Data Augmentation) </h3>

# In[21]:


#min_scale: defines how much of each image is to be selected minimum each time
# unique=True: to repeat same image with different version of the RandomResizedCrop transform

harsit_ankit = harsit_ankit.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = harsit_ankit.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)


# In[55]:


#To tell fastai we want to use these transforms on a batch, we use the batch_tfms parameter
# mul=2 means double the amount of augmentation compared to default

harsit_ankit = harsit_ankit.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = harsit_ankit.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)


# In[97]:


# to train our model, we’ll use RandomResizedCrop, an image size of 224 pixels, which is fairly standard for ...
# image classification, and the default aug_transforms

harsit_ankit = harsit_ankit.new(
 item_tfms=RandomResizedCrop(224, min_scale=0.5),
 batch_tfms=aug_transforms())
dls = harsit_ankit.dataloaders(path)


# <h2> Creating Learner and Fine-tuning </h2>

# In[57]:


learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)


# ---

# <h2> Confusion Matrix: Calculated using Validation Set </h2>

# In[58]:


# Confusion matrix tells us if our model is classifying one image as other image's label and vice versa 
# in two image-type dataset. Diagonal of matrix should be dark blue whereas other cells in white.


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[59]:


interp.plot_top_losses(5, nrows=1)


# ---

# <h2> Data Cleaning: ImageClassifierCleaner GUI </h2>

# In[99]:


# allows us to choose a category and the training versus validation set and view the...
# highest-loss images (in order), along with menus to allow images to be selected for...
# removal or relabeling:

cleaner = ImageClassifierCleaner(learn)
cleaner


# ---

# <h2> Turning model into Online Application </h2>

# <b> To save both parts of a model i.e. architecture and parameter, use the export method. It is done so that if we upload the model on a server and it is to be loaded and used, then we can be sure to have both the matching architecture and parameter. </b>

# In[60]:


# a file called export.pkl is saved

learn.export()


# In[61]:


# checking if the file exists using ls method

path = Path()
path.ls(file_exts='.pkl')


# In[62]:


# to create our inference learner from the exported file, we use load_learner

learn_inf = load_learner(path/'export.pkl')


# In[63]:


# vocab of the DataLoaders; that is, the stored list of all possible categories

learn_inf.dls.vocab


# <h2> Creating application in Jupyter notebook using Ipython widgets </h2>

# In[70]:


# File Upload widget

btn_upload = widgets.FileUpload()
btn_upload


# In[83]:


# grabbing the image

img = PILImage.create(btn_upload.data[-1])


# In[84]:


# output widget to display the image

out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl


# In[85]:


# getting the predictions

pred,pred_idx,probs = learn_inf.predict(img)


# In[86]:


# using label to display the predictions

lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred


# In[74]:


#creating classify button

btn_run = widgets.Button(description='Classify')
btn_run


# <h2> Creating Vertical GUI for Uploading image and classifying it: using previous functions </h2>

# In[87]:


# creating click event-handler

def on_click_classify(change):
 img = PILImage.create(btn_upload.data[-1])
 out_pl.clear_output()
 with out_pl: display(img.to_thumb(128,128))
 pred,pred_idx,probs = learn_inf.predict(img)
 lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


# In[88]:


btn_run.on_click(on_click_classify)


# In[89]:


# putting everything in the vertical box

VBox([widgets.Label('Select your image!'),
 btn_upload, btn_run, out_pl, lbl_pred])


# In[ ]:




