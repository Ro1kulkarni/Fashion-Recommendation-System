import streamlit as st  # Import the Streamlit library 
import os     #It is used for interacting with the opreting system, such as file handling directory operations.
from PIL import Image   # Import the Image module from the Python Imaging Library (PIL)
import numpy as np   #It is used for numerical computaions and array manipulation.
from numpy.linalg import norm    #The 'norm' function from Numpy's linear algebra modul for vector normalization.
import pickle   #It is used for serializing ans deserializing Python objects.
import tensorflow   #It is used for building & traning deep learming models.
from tensorflow.keras.preprocessing import image    #It is used for loading and preprocessing images.
from tensorflow.keras.layers import GlobalMaxPooling2D    #It is used for spatial data reduction in convolution neural networks.
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input #It is used for preprocess_input function for image preprocessing.
from sklearn.neighbors import NearestNeighbors   # Import NearestNeighbors class from sklearn.neighbors module

model = ResNet50(weights = "imagenet",include_top = False,input_shape = (224,224,3))
model.trainable = False 

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Specify the directory where the files are saved
save_dir = "D:/Fashion Recommendation System"

# Load feature_list from embeddings.pkl
with open(os.path.join(save_dir, "embeddings.pkl"), 'rb') as embeddings_file:
    feature_list = pickle.load(embeddings_file)

# Load filenames from filenames.pkl
with open(os.path.join(save_dir, "filenames.pkl"), 'rb') as filenames_file:
    filenames = pickle.load(filenames_file)

# Streamlit app title
st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
    
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))  # Resizing the image to (224, 224)
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)

    distannces,indices = neighbors.kneighbors([features])
    return indices

#file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)     #Display the file/image
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)
        
        # Show the recommended images
        with col1:
            # Load and display the first recommended image
            image_path_1 = filenames[indices[0][0]]
            image_1 = Image.open(image_path_1)
            st.image(image_1, caption='Recommended Image 1', use_column_width=True)

        with col2:
            # Load and display the second recommended image
            image_path_2 = filenames[indices[0][1]]
            image_2 = Image.open(image_path_2)
            st.image(image_2, caption='Recommended Image 2', use_column_width=True)

        with col3:
            # Load and display the third recommended image
            image_path_3 = filenames[indices[0][2]]
            image_3 = Image.open(image_path_3)
            st.image(image_3, caption='Recommended Image 3', use_column_width=True)
        
        with col4:
            # Load and display the third recommended image
            image_path_4 = filenames[indices[0][3]]
            image_4 = Image.open(image_path_4)
            st.image(image_4, caption='Recommended Image 4', use_column_width=True)
        
        with col5:
            # Load and display the third recommended image
            image_path_5 = filenames[indices[0][4]]
            image_5 = Image.open(image_path_5)
            st.image(image_5, caption='Recommended Image 5', use_column_width=True)

    else:
        st.header("Some error occured in file upload")
