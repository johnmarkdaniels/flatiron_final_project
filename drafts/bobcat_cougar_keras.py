


import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from keras import models
from keras import layers
from sklearn.metrics import confusion_matrix, f1_score
np.random.seed(123)
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[2]:


# Set up data into train and test directories with folders of bobcat and not_bobcat
train_data_dir = '/Users/j.markdaniels/Downloads/final_proj_data/bobcat_cougar_data/train/'#348 bobcat, 340 not_bobcat
test_data_dir = '/Users/j.markdaniels/Downloads/final_proj_data/bobcat_cougar_data/test/'#348 bobcat, 340 not_bobcat

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_data_dir, 
        target_size=(128, 128), batch_size=340)

train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_data_dir, 
        target_size=(128, 128), batch_size=340)

# create the data sets
train_images, train_labels = next(train_generator)
test_images, test_labels = next(test_generator)


# In[3]:


# get all the data in the directory split/test, and reshape them
data_te = ImageDataGenerator(rescale=1./255).flow_from_directory( 
        '/Users/j.markdaniels/Downloads/final_proj_data/bobcat_cougar_data/test/', 
        target_size=(224, 224), 
        batch_size = 340, 
        seed = 123)


# In[4]:


data_tr = ImageDataGenerator(rescale=1./255).flow_from_directory( 
        '/Users/j.markdaniels/Downloads/final_proj_data/bobcat_cougar_data/train/', 
        target_size=(224, 224), 
        batch_size = 340, 
        seed = 123) 


# In[5]:


#split images and labels
images_tr, labels_tr = next(data_tr)


# In[6]:


#split images and labels
images_te, labels_te = next(data_te)


# In[7]:


images = np.concatenate((images_tr, images_te))


# In[8]:


labels = np.concatenate((labels_tr[:,0], labels_te[:,0]))


# In[10]:


from sklearn.model_selection import train_test_split
X_model, X_test, y_model, y_test = train_test_split(images, labels, test_size=0.20, random_state=123)


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size=0.20, random_state=123)


# In[12]:


cnn = models.Sequential()
# cnn.add(layers.Conv2D(64, (1, 1), activation='relu', input_shape=(224, 224,  3)))
# cnn.add(layers.BatchNormalization())
# cnn.add(layers.MaxPooling2D((2, 2)))
# cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
# cnn.add(layers.BatchNormalization())
# # 64 bias parameters
# # 64 * (3 * 3 * 3) weight parametrs
# # Output is 64*224*224
# cnn.add(layers.MaxPooling2D((2, 2)))
# Output is 64*112*112
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224,  3)))
cnn.add(layers.BatchNormalization())
# 32 bias parameters
# 32 * (3*3*64)
# Output is 32*112*112 
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Flatten())
cnn.add(layers.Dense(32, activation='relu'))
cnn.add(layers.Dense(1, activation='sigmoid'))

cnn.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['acc'])


# In[ ]:


cnn1 = cnn.fit(X_train,
                    y_train,
                    epochs=5,
                    batch_size=5,
                    validation_data=(X_val, y_val))


# In[ ]:


print(cnn.summary())


# In[ ]:

results_train = cnn.evaluate(X_train, y_train)
results_test = cnn.evaluate(X_test, y_test)
print(results_train, results_test)


# In[ ]:





# In[ ]:




