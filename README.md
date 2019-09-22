# Final Project
## Flatiron School - DC
## Immersive Data Science Bootcamp
### J. Mark Daniels

### Summary
This project utilizes a Kaggle Dataset of over 14 thousand images of Oregon Wildlife to build and test three neural networks to categorize wildlife images into their proper categories by species. One model is a basic Convolutional Neural Network working with two classes. The second model used Inception v3 to classify two classes, and the last used Inception v3 again on the complete dataset of 20 classes.  

## Business Understanding
### Problem Definition
This project creates a neural network capable of classifying images of wildlife commonly found in the state of Oregon to enable Department of Fish and Wildlife, citizens, conservationists, and wildlife biologists to compile a database of wildlife sightings. A second neural network enables further refinement capabilities focusing on a single species as opposed to the 20 classes included in the multi-class model.

### Success/Evaluation Criteria
This project will be deemed successful if the models can outperform random chance when classifying images. Further comparison was obtained by measuring accuracy and F1 scores. 

## Data Understanding
### Public Data
This project was created using a Kaggle Data set found at: https://www.kaggle.com/virtualdvid/oregon-wildlife. Due to the large size of the image files involved (>5GB) the dataset is not included in this repository.

## Data Preparation
All images were rescaled and resized to allow for analysis using a neural network. Raw data files were untouched otherwise. Files were then organized into a directory structure that allowed keras to process the files using the ImageDataGenerator function.

### Validation
The binary model was validated using accuracy and F1 scores. The multi-class model was validated using accuracy and top_k_categorical_accuracy.

## Modeling
### Baseline
The binary model should operate with an accuracy rate above 50%. For the multi-class model, the model must perform with an accuracy rate above 5% to beat random chance in guessing the species class of the pictured animal. 
### Models
The models for this project were coded in Google Colab and saved as a Jupyter Notebook included in this repository. Functions are imported utilizing .py files that are also included in the repository. Two binary models were constructed. A basic convolutional neural network failed to obtain the accuracy and F1 score attained by the subsequent pre-trained model based on an Inception network. The Inception model was then expanded to encompass the entirety of the 20-class dataset.

## Deployment
### Relationship to Business Understanding
The current project far exceeds random chance in determining the species of animals pictured in the dataset images. It provides a convenient means of classifying and assisting in the reporting of wildlife in the state of Oregon.
### Web Presence
With a suitable front end, these models are prepared for incorporation into the Oregon Department of Fish and Wildlife website to supplement existing wildlife reporting forms and tracking systems.

## Next Steps
### Model Improvement
Retrain models supplementing with additional trail camera images taken in low-light and night images.
Utilize feature engineering or model improvements to improve species differentiation scores on commonly confused species.
### Product Roadmap
Build web interface to allow Oregonians to upload images
Extract geotagging information from photos and plot to maps in real-time.
Track annual trends in migration and population.
