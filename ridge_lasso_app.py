import streamlit as st
import numpy as np
import pickle
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

# ---------- CONFIGURE PAGE THEME ------------
st.set_page_config(
    page_title="Diabetes ML Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------- SIDEBAR ------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3875/3875432.png", width=100)
    st.title("🔬 Diabetes ML App")
    st.markdown("""
    This app predicts diabetes progression  
    using advanced Ridge & Lasso models.
    
    * Powered by **Machine Learning** and  
      **MongoDB** database storage!
    """)
    st.markdown("---")
    st.info("✨ Enter data, select a model, and save results automagically!")

    st.write("#### 📦 Project Details")
    st.markdown("""
    - **Tech:** Streamlit Web UI  
    - **Models:** Ridge & Lasso (.pkl)  
    - **Database:** MongoDB Atlas  
    - **Deployment:** GitHub + Streamlit Cloud  
    - **Responsive:** Works on PC, tablet, and phone
    """)
    st.markdown("---")
    st.write("Made with 🩺, 🚀 and ❤️ by Kamran Sohail")

# ---------- MongoDB connection setup -------------
uri = "mongodb+srv://Kamran:kamran1234@cluster0.age032x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Machine_Learning']
collection = db["Ridge_Lasso_Model"]

# ------------- Model loaders ---------------------
@st.cache_resource(show_spinner=False)
def load_ridge_model():
    try:
        with open('ridge_diabetes_model.pkl', 'rb') as f:
            ridge_model = pickle.load(f)
        return ridge_model
    except Exception as e:
        st.error(f"❌ Error loading Ridge model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_lasso_model():
    try:
        with open('lasso_diabetes_model.pkl', 'rb') as f:
            lasso_model = pickle.load(f)
        return lasso_model
    except Exception as e:
        st.error(f"❌ Error loading Lasso model: {e}")
        return None

# ------------ Colorful main UI -------------------
st.markdown("<h1 style='text-align: center; color: #0074D9;'>🩺 Diabetes Progression Machine Learning Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #FF4136; text-align: center;'>Enter your health data, pick your model, and get a prediction with style! 🎨</h4>", unsafe_allow_html=True)
st.markdown("---")

feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
tooltips = [
    "Patient's age in years",
    "Sex (male:1, female:0)",
    "Body Mass Index (BMI, standardized)",
    "Average Blood Pressure",
    "S1: TC/HDL ratio",
    "S2: LDL/HDL ratio",
    "S3: TCH/HDL ratio",
    "S4: Serum 4",
    "S5: Serum 5",
    "S6: Serum 6"
]

with st.form(key="prediction-form"):
    st.subheader("📄 Enter your values (Touch/Click-friendly, Mobile-Optimized):")
    input_values = []
    # Responsive columns for better mobile and PC experience
    cols = st.columns([1, 1])  # two equal-width columns
    for idx, name in enumerate(feature_names):
        col = cols[idx % 2]
        val = col.number_input(
            f"{name.upper()}",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.01,
            help=tooltips[idx],
            format="%.6f"
        )
        input_values.append(val)

    st.markdown("---")
    model_type = st.radio(
        "🧮 Choose Regression Model:",
        ("Ridge Regression", "Lasso Regression"),
        help="Select which model to use for prediction."
    )

    submit_btn = st.form_submit_button("🚀 Predict My Diabetes Progression")

if submit_btn:
    input_array = np.array([input_values])
    # Cache usage makes UI and prediction faster, even on mobile/tablet
    model = load_ridge_model() if model_type == "Ridge Regression" else load_lasso_model()
    if model is None:
        st.error("❌ Model could not be loaded. Check your .pkl files!")
    else:
        try:
            prediction = model.predict(input_array)
            st.balloons()
            st.markdown(
                f"<div style='background-color:#8EE4AF;padding:18px;border-radius:10px;text-align:center;'>"
                f"🔮 <b>Prediction for diabetes progression:</b> <span style='color:#FF4136;font-size:24px'>{prediction[0]:.4f}</span>"
                f"</div>", unsafe_allow_html=True
            )

            # Prepare record for MongoDB
            record = {feature_names[i]: input_values[i] for i in range(10)}
            record["ModelUsed"] = model_type
            record["Prediction"] = float(prediction[0])
            collection.insert_one(record)
            st.success("🗄️ Prediction & input saved to database!")

            st.markdown("---")
            st.subheader("📈 Recent Predictions (last 5):")
            try:
                data = pd.DataFrame(list(collection.find().sort("_id", -1).limit(5)))
                if len(data) > 0:
                    data = data.drop(columns='_id')
                    # Use wide container for mobile-friendly table
                    st.dataframe(data, use_container_width=True)
                else:
                    st.info("No previous predictions found.")
            except Exception as error:
                st.error(f"Could not fetch previous predictions: {error}")
        except Exception as e:
            st.error(f"❌ Error during prediction or saving to database: {e}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#0074D9;font-size:18px;'>
    <b>Project deployed and maintained via GitHub + Streamlit Cloud for lightning-fast cross-device performance.<br>
    Made with 🩺, 🚀 and ❤️ by Kamran Sohail</b>
    </div>
    """, unsafe_allow_html=True
)
