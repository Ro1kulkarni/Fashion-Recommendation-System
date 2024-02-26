import tensorflow    #It is used for building & traning deep learming models. 
from tensorflow.keras.preprocessing import image    #It is used for loading and preprocessing images.
from tensorflow.keras.layers import GlobalMaxPooling2D    #It is used for spatial data reduction in convolution neural networks.
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input #It is used for preprocess_input function for image preprocessing.
import numpy as np          #It is used for numerical computaions and array manipulation.
from numpy.linalg import norm        #The 'norm' function from Numpy's linear algebra modul for vector normalization.
import os          #It is used for interacting with the opreting system, such as file handling directory operations. 
from tqdm import tqdm        #It is used for displaying progress bars during iterations.
import pickle           #It is used for serializing ans deserializing Python objects.

model = ResNet50(weights = "imagenet",include_top = False,input_shape = (224,224,3))
model.trainable = False 

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Resizing the image to (224, 224)
    img_array = image.img_to_array(img)                     # Convert the image to a numpy array
    expanded_img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension to the image array to make it suitable for model input
    preprocessed_img = preprocess_input(expanded_img_array) # Preprocess the input image according to the requirements of the model
    result = model.predict(preprocessed_img).flatten()      # Use the pre-trained model to extract features from the preprocessed image 
    normalized_result = result / np.linalg.norm(result)     # Normalize the extracted features to ensure they have unit norm
    return normalized_result                                # Return the normalized feature vector

filenames = []
images = 'D:\images'

for file in os.listdir(images):
    filenames.append(os.path.join(images,file))

print("Number of Images =",len(filenames))
print("Path of first five images =",filenames[0:5])

feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open("embeddings.pkl",'wd'))
pickle.dump(filenames,open("filnames.pkl",'wb'))

# Specify the directory Path
save_dir = "D:/"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save feature_list to embeddings.pkl in the specified directory
pickle.dump(feature_list, open(os.path.join(save_dir, "embeddings.pkl"), 'wb'))

# Save filenames to filenames.pkl in the specified directory
pickle.dump(filenames, open(os.path.join(save_dir, "filenames.pkl"), 'wb'))


print("Shape for feature_list =",np.array(feature_list).shape)

