import streamlit as st
import pandas as pd 
import numpy as np 
import joblib 
from sklearn.preprocessing import StandardScaler
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://Anurag:Anurag1234@cluster0.bagh3cm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["Iris"]
collection = db["iris prediction"]

# Writing the function to load the model 
def load_model():
        model = joblib.load("svm_multi.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    
# Writing a function where user fill their choice of data and then get converted into our transformed data 
def preprocessing_input_data(data, scaler):
    # Create DataFrame with proper column names and values
    df = pd.DataFrame([data])
    df_transform = scaler.transform(df)
    return df_transform

# Writing a predict function 
def predict_data(data):
    lasso_model, scaler = load_model()
    process_data = preprocessing_input_data(data, scaler)
    prediction = lasso_model.predict(process_data)
    return prediction

# Creating the UI for app 
def main():
    st.title("Iris prediction")
    st.write("Enter your data to get a prediction for 0--> setosa , 1--> versicolor , 2--> varginica")

    # Now we are going to create fields where user can fill the data
    sepallength = st.number_input("sepal length (cm)", min_value=0, max_value=100, value=1)
    sepalwidth = st.number_input("sepal width (cm)", min_value=0, max_value=100, value=1)  # Fixed: was "sex" instead of "BMI"
    petallength = st.number_input("petal length (cm)", min_value=0, max_value=100, value=1)
    petalwidth = st.number_input("petal width (cm)", min_value=0, max_value=100, value=1)

    # Now we are going to create a button at the bottom which when clicked sends all the user input into the model
    if st.button("Predict the iris"):
        # Fixed: Values and keys were swapped
        user_data = {
            "sepal length (cm)": sepallength,
            "sepal width (cm)": sepalwidth,
            "petal length (cm)": petallength,
            "petal width (cm)": petalwidth,
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is {prediction}")
        user_data["prediction"] = float(prediction)                                                                                                                            #user data me prediction data ko add kro 

        # Imagine you have a big box of toys (that's called user_data), and each toy has a name tag on it. But sometimes the name tags are written in a special computer language that's hard to read.
        # This code is like having a helpful friend who goes through your toy box and rewrites all the name tags in regular English so everyone can understand them better.
        user_data = {key: int(value) if isinstance(value , np.integer) else float(value) if isinstance(value , np.floating) else value for key , value in user_data.items()}   

    collection.insert_one(user_data)                                                                                                               
#To execute the above url function line 
if __name__ == "__main__" :
    main()