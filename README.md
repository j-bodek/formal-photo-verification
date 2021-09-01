# Formal Photo Verification App 

This easy web application predicts whether the photo is in a formal style. <br/>
APP: https://share.streamlit.io/liljack118/formal-photo-verification/app/app.py <br/>
API: https://formal-photo-verification-api.herokuapp.com/


## Data 
To train network I used dataset of 3500 images in total. <br/>
Training set - 2800 images ( 1200 positive / 1600 negative) <br/>
Validation set - 350 images (150 positive / 200 negative)<br/>
Test set - 350 images (150 positive / 200 negative) <br/>

Invalid images contains:
- rotated head 
- frontal head photos with sunglasses/hats/headset/unnatural mimic
- frontal head photos on wrong background
- animated frontal head photos
- empty rooms photos
- person with body 
 
Invalid images from every subfolder were distributed equally to training,validation and test set to make sure that images in every set came from same distribution.
  
