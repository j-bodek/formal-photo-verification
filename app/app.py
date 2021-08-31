import streamlit as st
import numpy as np 
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import Input
import requests
import seaborn as sns 
from io import BytesIO


class page_settings():
    choosen_threshold = None
    data_amount = None
    visualize_tsne = False
    img_arr = None
    images = []
    api_images = []
    probability = None
    index = None
    


class sidebar():
    def __init__ (self):

        st.sidebar.write("""# Model that will predict your photo
        ResNet50""")
        

        blank_space()

        page_settings.choosen_threshold = st.sidebar.slider("Choose a threshold", 0.0, 1.0, 0.5, step = 0.01)
        blank_space()


        page_settings.data_amount = st.sidebar.radio(
            'How many examples would you like to test?',
            ('Only One', 'Batch'))
        blank_space()


        if page_settings.data_amount == 'Batch':
            page_settings.visualize_tsne = st.sidebar.checkbox('Visualize data with T-sne')



class help_page():

    def __init__(self):
        st.write("""
        # When image is valid:
        - Eyes fully visible
        - Photo should show only head and shoulders
        - Plain light coloured background

        # When image is invalid:
        - Hat/head covered
        - Sunglasses
        - Background is not plain light coloured
        - Head is rotated
        - Shoulders are not square to the camera
        """)

        st.write('# Examples of invalid and valid images:')
        col1, col2, col3 = st.columns(3)
        col1.image('app/images/valid_1.jpg', caption = 'Valid',  use_column_width=True)
        col2.image('app/images/valid_2.jpg', caption = 'Valid', use_column_width=True)
        col3.image('app/images/invalid_1.jpg', caption = 'Invalid', use_column_width=True)
        col1.image('app/images/invalid_2.jfif', caption = 'Invalid', use_column_width=True)
        col2.image('app/images/invalid_3.jfif', caption = 'Invalid', use_column_width=True)
        col3.image('app/images/invalid_4.png', caption = 'Invalid', use_column_width=True)



class content_page():

    image_options = ['Invalid', 'Valid']

    def _blank_space(self):
        st.write('\n')
        st.write('\n')

    def _convert_to_bytes(self, files):
        images = []
        if type(files) != list:
            img = Image.open(files)
            page_settings.images.append(img)

            output = BytesIO()
            img.save(output, 'bmp')
            images.append(output.getvalue())
        
        else:
            for file in files:
                img = Image.open(file)
                page_settings.images.append(img.resize((200,200), Image.ANTIALIAS))

                output = BytesIO()
                img.save(output, 'bmp')
                images.append(output.getvalue())

        return images
        

    def _files_list(self, imgs_list):
        files = []
        if type(imgs_list) != list:
            item = ('files', (imgs_list))
            files.append(item)
        
        else:
            for item in imgs_list:
                item = ('files', (item))
                files.append(item)
        
        return files

    def _predict(self, files):
        r = requests.post('https://formal-photo-verification-api.herokuapp.com/api/predict', files=files)
        data = r.json()
        predictions = data["items"]

        return predictions


    def _index_and_probability(self, predictions, threshold):

        predictions = np.array(predictions)

        index = (predictions[:,1] >= threshold).astype(int)

        # calculate percentage probability
        probability = np.choose(index, predictions.T) * 100

        return index, probability


    # Function which convert image to numpy array 
    def _imgs_arr(self, imgs):

        imgs_arr = np.empty((0,224,224,3), int)

        for img in imgs:
            #convert image to rgb
            img = img.convert('RGB')

            #resize image to 224*224
            img_arr = tf.image.resize(img, (224,224))

            img_arr = image.img_to_array(img_arr) 

            # expend one dimension couse we want array of shape (1,224,224,3) instead of (224,224,3)
            img_arr = np.expand_dims(img_arr, axis = 0)

            #devide every value in array by 255 (so we get values between 0 and 1)
            img_arr = img_arr / 255.

            imgs_arr = np.append(imgs_arr, img_arr, axis = 0)

        return imgs_arr




class one_img_page(content_page):

    def __init__(self):

        self.file = st.file_uploader('Pick a file', type = ('TIFF','JFIF','JPG','JPEG','PNG'))

        self.make_prediction = st.button('PREDICT!')

    
    def show_data(self):

        if self.file is not None and self.make_prediction:
            
            page_settings.api_images = self._convert_to_bytes(self.file)

            self._blank_space()

            #create files list for api
            files = self._files_list(page_settings.api_images)
            
            #calculate predictions
            predictions = self._predict(files)

            page_settings.index, page_settings.probability = self._index_and_probability(predictions, page_settings.choosen_threshold)

            # Display result
            st.write('# Image predicted as ' + self.image_options[page_settings.index[0]] + '!')

            st.write('Probability ' + format(page_settings.probability[0], ".2f") + '%')

            st.image(page_settings.images, use_column_width=True)







class multiple_images(content_page):

    def __init__(self):

        self.files = st.file_uploader('Pick files', type = ('TIFF','JFIF','JPG','JPEG','PNG'), accept_multiple_files=True)

        self.make_prediction = st.button('PREDICT!')

        # np empty array to store arrays of all choosen images
        self.imgs_arr = None




    def show_data(self):

        # Start predicting only if two or more files are choosen and prediction button was clicked
        if len(self.files) >= 2 and self.make_prediction:
            #Predict probabilities and indexes
            page_settings.api_images = self._convert_to_bytes(self.files)

            #create files list for api
            files = self._files_list(page_settings.api_images)
            
            #calculate predictions
            predictions = self._predict(files)

            page_settings.index, page_settings.probability = self._index_and_probability(predictions, page_settings.choosen_threshold)
            
            #Display results
            self._display_images()
            #If choosen display tsne visualization
            if page_settings.visualize_tsne:
                self.imgs_arr = self._imgs_arr(page_settings.images)
                self._display_tsne()
        
    

    def _display_images(self):

        #create three columns
        col1, col2, col3 = st.columns(3)
        #start from column 1
        col = 1

        # loop through three lists at the same time
        for prob,option,image in zip(page_settings.probability, page_settings.index, page_settings.images):

            #display results in coresponding columns
            if col % 2 == 0:
                col2.image(image, use_column_width=True)
                col2.write('Image predicted as ' + self.image_options[option] + '!')
                col2.write('Probability ' + format(prob, ".2f") + '%')
                col2.write('\n')
            elif col % 3 == 0:
                col3.image(image, use_column_width=True)
                col3.write('Image predicted as ' + self.image_options[option] + '!')
                col3.write('Probability ' + format(prob, ".2f") + '%')
                col3.write('\n')
            else:
                col1.image(image, use_column_width=True)
                col1.write('Image predicted as ' + self.image_options[option] + '!')
                col1.write('Probability ' + format(prob, ".2f") + '%')
                col1.write('\n')

            # to eliminate situation when "col" is divisible by 2 and 3 (for example 6 should be ordered to 
            # third column but will be ordered to column number two) set col to 1 if it will be equal to 3
            col = col + 1 if col < 3 else 1

    
    def _display_tsne(self):
        #specify input shape
        input_tensor = Input(shape=(224, 224, 3))
        #load head-less model from keras (with average pooling to get (None, 2048) shape instead of (None,2,2,2048))
        base_model = ResNet152V2(include_top=False, input_tensor=input_tensor, weights='imagenet', pooling='avg', classes = 2)

        #make predictions using base model
        tsne_features = base_model.predict(self.imgs_arr)
        #use tsne keras function
        tsne = TSNE(n_components=2, verbose=1, perplexity=40,learning_rate = 50, n_iter=600)
        tsne_results = tsne.fit_transform(tsne_features)

        #plot tsne results
        tsne_res = np.vstack((tsne_results.T, page_settings.index)).T
        tsne_df = pd.DataFrame(data = tsne_res, columns = ('X', 'Y', 'label'))
        fig = sns.FacetGrid(tsne_df, hue='label', height = 6).map(plt.scatter, 'X', 'Y', alpha = .7).add_legend()
        st.pyplot(fig)






#WRITE BLANK LINE (MARGIN)

def blank_space():
    st.sidebar.write('\n')
    st.sidebar.write('\n')




sidebar()

help_page()

if page_settings.data_amount == 'Only One':
    page_settings.start = True
    one_page = one_img_page()
    one_page.show_data()


if page_settings.data_amount == 'Batch':
    page_settings.start = True
    multiple_images_page = multiple_images()
    multiple_images_page.show_data()

