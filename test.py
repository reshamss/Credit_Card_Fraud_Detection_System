import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import base64

# Load and prepare the data
data = pd.read_csv('creditcard.csv')

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

X = balanced_data.drop('Class', axis=1)
Y = balanced_data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression(max_iter=200)  # Ensure convergence
model.fit(X_train, Y_train)

train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

# Streamlit app

def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .title {{
            color: white !important;
        }}
        .stTextInput > div {{
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 5px;
        }}
          .stTextInput > label {{
            color: white !important;
        }}
        .stTextInput > div > input {{
            color: white !important;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            margin-top: 10px;
            background: transparent !important;
        }}
        .stButton > button {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        .stButton > button:hover {{
            background-color: #45a049;
        }}
        .prediction {{
            color: #d3d3d3 !important;  /* Dark white color */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path
add_bg_from_local("cardimg.jpg")

# Your Streamlit code
st.markdown("<h1 class='title'>Credit Card Fraud Detection</h1>", unsafe_allow_html=True)

# Input features with custom CSS
input_df = st.text_input("Input all features ")

# Submit button with custom CSS
submit = st.button("Submit")

if submit:
    try:
        # Process input and make prediction
        input_df_lst = input_df.split(',')
        features = np.array([float(item.replace('"', '').strip()) for item in input_df_lst], dtype=np.float64)
        
        if features.shape[0] != X.shape[1]:
            st.write(f"Error: Expected {X.shape[1]} features, but got {features.shape[0]}. Please ensure you input the correct number of features.")
        else:
            prediction = model.predict(features.reshape(1, -1))
            if prediction[0] == 0:
                st.markdown("<p class='prediction'>Prediction: Legitimate Transaction</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='prediction'>Prediction: Fraudulent Transaction</p>", unsafe_allow_html=True)
    except ValueError :
        st.write(" Please ensure all inputs are numerical and comma-separated.")
       
