# -*- coding: utf-8 -*-


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    tab = st.sidebar.selectbox('Section', ('Visualisation','Model Prediction'))
    if tab=='Visualisation':
        html_temp = """
        	<div style="background-color:tomato;padding:10px">
        	<h2 style="color:white;text-align:center;">Visualisation</h2>
        	</div>
        	"""
        st.markdown(html_temp,unsafe_allow_html=True)
        @st.cache
        def load_df():
            df = pd.read_csv('PreprocessedDataframe.csv')
            return df
        
        data_load_state = st.text('Loading data...')
        df = load_df()
        data_load_state.text("Data loading done!")
        
        
        if st.checkbox('Show dataframe'):
            st.write(df)
            
        st.subheader('Filter Columns')
        colWan = st.multiselect('Which variables do you want to see?', df.columns)
        new_df = df[colWan]
        st.write(new_df)
            
        st.subheader('Visualization 1 (Count plot on 1 variable excluding Date and Time)')
        colV1 = st.selectbox('Which feature on x?', df.columns[2:],key='First')
        
        st.bar_chart(df[colV1].value_counts())
        
        st.subheader('Visualization 2 (Count plot on 2 variables excluding Date and Time)')
        colXV2 = st.selectbox('Which feature on x?', df.columns[2:],key='Sec')
        colYV2 = st.selectbox('Which feature to group by?', df.columns[2:])
        
        if(colXV2 and colYV2):
            fig= plt.figure()
            ax = fig.add_subplot(111)
            ax = sns.countplot(x=colXV2,data= df,hue =df[colYV2])
            ax.legend(prop=dict(size=4))
            st.pyplot(fig)
            
    elif tab=='Model Prediction':
        with open('encodersDict', 'rb') as file:
            loaded_encoders = pickle.load(file)
    
        with open('GradientBoosting_Specs.pkl', 'rb') as file:
            gb_specs_pkl = pickle.load(file)
        
        with open('CategoricalNB_Washer.pkl', 'rb') as file:
            chi2_catNB_washer = pickle.load(file)
        
        with open('KNN_Dryer.pkl', 'rb') as file:
            boruta_knn_dryer_pkl = pickle.load(file)
        
        #@app.route('/')
        def welcome():
        	return "Welcome All"
        
        def predict_specs(hourCat, pantsCol, washer, ageGroup, bodySize, withKids, basketCol, shirtType, washItem, race, pantsType, basketSize, dryer, shirtCol):
        	spec_columns = ['HOUR_CATEGORY', 'PANTS_COLOUR', 'WASHER_NO', 'AGE_GROUP', 'BODY_SIZE', 'WITH_KIDS', 'BASKET_COLOUR', 'SHIRT_TYPE', 'WASH_ITEM', 'RACE', 'PANTS_TYPE', 'BASKET_SIZE', 'DRYER_NO', 'SHIRT_COLOUR']
        	given_row = pd.DataFrame([[hourCat, pantsCol, washer, ageGroup, bodySize, withKids, basketCol, shirtType, washItem, race, pantsType, basketSize, dryer, shirtCol]], columns = spec_columns)
        
        	for col in given_row.columns: 
        		if col not in ['DRYER_NO', 'WASHER_NO']:
        			given_row[col] = loaded_encoders[col].transform(given_row[col])
        			
        	print(given_row.values)
        	prediction_encoded = gb_specs_pkl.predict(given_row.values)
        
        	prediction_decoded = pd.DataFrame(prediction_encoded, columns=['SPECTACLES'])
        	
        	for col in prediction_decoded.columns: 
        		prediction_decoded[col] = loaded_encoders[col].inverse_transform(prediction_decoded[col])
        		
        	print(prediction_decoded.values)
        	return prediction_decoded.values
        
        def predict_washer(pantsCol, attire, shirtCol, specs, basketCol, kidCat, bodySize, ageGroup, withKids, basketSize, pantsType, gender, race, hourCat):
        	washer_columns = ['PANTS_COLOUR', 'ATTIRE', 'SHIRT_COLOUR', 'SPECTACLES', 'BASKET_COLOUR', 'KIDS_CATEGORY', 'BODY_SIZE', 'AGE_GROUP', 'WITH_KIDS', 'BASKET_SIZE', 'PANTS_TYPE', 'GENDER', 'RACE', 'HOUR_CATEGORY']
        	given_row = pd.DataFrame([[pantsCol, attire, shirtCol, specs, basketCol, kidCat, bodySize, ageGroup, withKids, basketSize, pantsType, gender, race, hourCat]], columns = washer_columns)
        
        	for col in given_row.columns: 
        		if col not in ['DRYER_NO', 'WASHER_NO']:
        			given_row[col] = loaded_encoders[col].transform(given_row[col])
        
        	print(given_row.values)
        	prediction = chi2_catNB_washer.predict(given_row.values)
        
        	print(prediction)
        	return prediction
        
        def predict_dryer(shirtCol, basketCol, pantsCol, ageGroup, bodySize, race, washItem, hourCat, pantsType, kidCat, specs, attire, gender, basketSize):
        	dryer_columns = ['SHIRT_COLOUR', 'BASKET_COLOUR', 'PANTS_COLOUR', 'AGE_GROUP', 'BODY_SIZE', 'RACE', 'WASH_ITEM', 'HOUR_CATEGORY', 'PANTS_TYPE', 'KIDS_CATEGORY', 'SPECTACLES', 'ATTIRE', 'GENDER', 'BASKET_SIZE']	
        
        	given_row = pd.DataFrame([[shirtCol, basketCol, pantsCol, ageGroup, bodySize, race, washItem, hourCat, pantsType, kidCat, specs, attire, gender, basketSize]], columns = dryer_columns)
        
        	for col in given_row.columns: 
        		if col not in ['DRYER_NO', 'WASHER_NO']:
        			given_row[col] = loaded_encoders[col].transform(given_row[col])
        
        	print(given_row.values)
        	prediction = boruta_knn_dryer_pkl.predict(given_row.values)
        
        	print(prediction)
        	return prediction
        
       	html_temp = """
       	<div style="background-color:tomato;padding:10px">
       	<h2 style="color:white;text-align:center;">Prediction of Spectacles, Washer No, Dryer No</h2>
       	</div>
       	"""
       	st.markdown(html_temp,unsafe_allow_html=True)
       	#select target variables
       	target_variables = st.radio("Target Variable",('spectacles', 'washer_no', 'dryer_no'))
       
       	#common features among 3 classifiers
       	race = st.radio("Race",('chinese', 'foreigner', 'indian', 'malay','Unknown'))
       	ageGroup= st.radio("Age Group",('28-34', '35-41', '42-48', '49-55', 'Unknown'))
       	bodySize = st.radio("Body Size",('fat', 'moderate', 'thin','Unknown'))
       	basketSize = st.radio("Basket Size",('big', 'small','Unknown'))
       	basketCol = st.radio("Basket Color",('black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow','Unknown' ))
       	pantsType = st.radio("Pants Type",( 'long', 'short','Unknown'))
       	pantsCol = st.radio("Pants Colour",('black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'))
       	shirtCol = st.radio("Shirt Colour",('black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow','Unknown'))
       	hourCat = st.radio("Hour Category",('0-5', '6-11','12-17', '18-23'))
       
       
       	
       
       
       	#SPEC features
       	if target_variables == "spectacles":
       		shirtType = st.radio("Shirt Type",('longsleeve', 'short_sleeve'))
       		washer= st.radio("Washer No",(3, 4, 5, 6))
       		dryer = st.radio("Dryer No",(7, 8, 9, 10))
       		washItem = st.radio("Wash Item",('blankets', 'clothes','Unknown'))
       		withKids = st.radio("With Kids",('no', 'yes'))
       	elif target_variables == "washer_no" or target_variables == "dryer_no":#washer and dryer common features
       		attire = st.radio("Attire",('casual', 'formal', 'traditional'))
       		specs = st.radio("Spectacles",('no', 'yes'))
       		gender = st.radio("Gender",('female', 'male'))
       		kidCat = st.radio("Kids Category",('baby', 'no_kids', 'toddler', 'young'))
       		if target_variables == "washer_no":
       			withKids = st.radio("With Kids",('no', 'yes'))
       		elif target_variables == "dryer_no":
       			washItem = st.radio("Wash Item",('blankets', 'clothes','Unknown'))
       
       
    
    
       	
       
       	result=""
       	#ordering of column follow training set columns
       	if st.button("Predict"):
       		if target_variables == "spectacles":
       			result = predict_specs(hourCat, pantsCol, washer, ageGroup, bodySize, withKids, basketCol, shirtType, washItem, race, pantsType, basketSize, dryer, shirtCol)
       		elif  target_variables == "washer_no":
       			result = predict_washer(pantsCol, attire, shirtCol, specs, basketCol, kidCat, bodySize, ageGroup, withKids, basketSize, pantsType, gender, race, hourCat)
       		elif  target_variables == "dryer_no":
       			result = predict_dryer(shirtCol, basketCol, pantsCol, ageGroup, bodySize, race, washItem, hourCat, pantsType, kidCat, specs, attire, gender, basketSize)
       
       		st.success('The output is {}'.format(result))
      
    
if __name__=='__main__':
    main()
        
