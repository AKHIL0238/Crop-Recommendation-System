import streamlit as st
import numpy as np
import pickle
import os
import requests

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱")

@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        sc = pickle.load(open('standscaler.pkl', 'rb'))
        ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
        return model, sc, ms
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

def predict_crop(features, model, sc, ms):
    """Make prediction using the pre-trained model"""
    try:
        single_pred = np.array(features).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)
        
        if prediction[0] in crop_dict:
            return crop_dict[prediction[0]]
        return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def ai_recommendations(crop, features, chat_input=None, chat_history=None):
    """Fetch cultivation insights from Mistral Nemo model"""
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407"
    api_token = "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"

    base_prompt = f"""Provide detailed agricultural guidance for {crop} cultivation, 
    focusing on:
    1. Optimal cultivation process
    2. Recommended fertilizers
    3. Pest prevention strategies
    4. Best cultivation seasons
    5. Key growth requirements"""

    if chat_history:
        history_context = "\n\nPrevious Conversation Context:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        prompt = f"{base_prompt}\n{history_context}"
    else:
        prompt = base_prompt

    detailed_prompt = f"""{base_prompt}

    Detailed Soil and Environmental Parameters:
    - Nitrogen: {features[0]:.1f}
    - Phosphorus: {features[1]:.1f}
    - Potassium: {features[2]:.1f}
    - Temperature: {features[3]:.1f}°C
    - Humidity: {features[4]:.1f}%
    - pH: {features[5]:.1f}
    - Rainfall: {features[6]:.1f} mm

    Provide comprehensive agricultural insights taking these specific parameters into account."""

    if chat_input:
        detailed_prompt += f"\n\nLatest User Query: {chat_input}"

    headers = {"Authorization": f"Bearer {api_token}"}
    
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": detailed_prompt})
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return "Unable to fetch agricultural insights."
    except Exception as e:
        return f"Error fetching insights: {str(e)}"

def main():
    st.title("Crop Recommendation System 🌱")
    st.write("Welcome to the crop recommendation system! Please enter the required parameters:")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_crop' not in st.session_state:
        st.session_state.current_crop = None
    if 'current_features' not in st.session_state:
        st.session_state.current_features = None

    model, sc, ms = load_models()
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            nitrogen = st.number_input("Nitrogen level (N)", min_value=0.0, max_value=140.0, step=1.0)
            phosphorus = st.number_input("Phosphorus level (P)", min_value=0.0, max_value=145.0, step=1.0)
            potassium = st.number_input("Potassium level (K)", min_value=0.0, max_value=205.0, step=1.0)
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)

        with col2:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
            ph = st.number_input("pH value", min_value=0.0, max_value=14.0, step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, step=0.1)

        if st.button("Get Recommendation"):
            feature_list = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
            result = predict_crop(feature_list, model, sc, ms)
            if result:
                st.session_state.chat_history = []
                st.session_state.current_crop = result
                st.session_state.current_features = feature_list
                st.success(f"Based on the parameters, {result} is the best crop to cultivate! 🌱")
                with st.spinner("Please wait,generating Agricultural insights..."):
                    description = ai_recommendations(result, feature_list)
                with st.expander(f"Agricultural Insights for {result}"):
                    st.write(description)
        if st.session_state.current_crop:
            st.subheader(f"Chat about {st.session_state.current_crop} Cultivation")
            for msg in st.session_state.chat_history:
                st.chat_message(msg['role']).write(msg['content'])
            chat_input = st.chat_input("Ask a specific question about crop cultivation")
            if chat_input:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': chat_input
                })
                st.chat_message('user').write(chat_input)
                with st.spinner("Generating response..."):
                    chat_response = ai_recommendations(
                        st.session_state.current_crop, 
                        st.session_state.current_features, 
                        chat_input, 
                        st.session_state.chat_history
                    )
                    st.session_state.chat_history.append({
                        'role': 'AI',
                        'content': chat_response
                    })
                    st.chat_message('AI').write(chat_response)

if __name__ == "__main__":
    main()