import tensorflow    #It is used for building & traning deep learming models. 
from tensorflow.keras.preprocessing import image    #It is used for loading and preprocessing images.
from tensorflow.keras.layers import GlobalMaxPooling2D    #It is used for spatial data reduction in convolution neural networks.
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input #It is used for preprocess_input function for image preprocessing.
import numpy as np          #It is used for numerical computaions and array manipulation.
from numpy.linalg import norm  #The 'norm' function from Numpy's linear algebra modul for vector normalization.
import pickle          #It is used for serializing ans deserializing Python objects.
import os           #It is used for interacting with the opreting system, such as file handling directory operations.

# Specify the directory where the files are saved
save_dir = "D:/Fashion Recommendation System"

# Load feature_list from embeddings.pkl
with open(os.path.join(save_dir, "embeddings.pkl"), 'rb') as embeddings_file:
    feature_list = pickle.load(embeddings_file)

# Load filenames from filenames.pkl
with open(os.path.join(save_dir, "filenames.pkl"), 'rb') as filenames_file:
    filenames = pickle.load(filenames_file)

model = ResNet50(weights = "imagenet",include_top = False,input_shape = (224,224,3))
model.trainable = False 

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

from sklearn.neighbors import NearestNeighbors

img = image.load_img("D:/Fashion Recommendation System/saari.jpg", target_size=(224, 224))  # Resizing the image to (224, 224)
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / np.linalg.norm(result)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Initialize NearestNeighbors with specified parameters
neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
# Fit NearestNeighbors to the feature_list
neighbors.fit(feature_list)


# Find the nearest neighbors of the normalized_result
distannces,indices = neighbors.kneighbors([normalized_result])

# Print the indices of nearest neighbor images
print(indices)
a=[]

# Loop through the indices of nearest neighbor images (excluding the first index which is the query image itself)
for file in indices[0][1:6]:
    a.append(filenames[file])
print(a)
# Convert backslashes to forward slashes
file_paths_forward_slash = [path.replace("\\", "/") for path in a]
print(file_paths_forward_slash)

for path in file_paths_forward_slash:
    # Load the image
    img = mpimg.imread(path)
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Hide axis
    plt.show()  
