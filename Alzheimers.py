import streamlit as st
import requests
import json

# Define the Streamlit app
st.title("Predictive Analytics for Alzheimer's Disease Onset")

st.markdown("""
This app uses the GPT-4 model from OpenAI to analyze risk factors and provide insights related to the onset of Alzheimer's Disease. 
Please input your OpenAI API key and details related to your risk factors.
""")

# Get OpenAI API key from user
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Input fields for various risk factors
age = st.number_input("Age", min_value=30, max_value=120, step=1)
gender = st.selectbox("Gender", ("Male", "Female"))
smoking = st.selectbox("Current Smoking Status", ("Non-Smoker", "Smoker"))
physical_activity = st.selectbox("Physical Activity Level", ("High", "Moderate", "Low"))
head_trauma = st.selectbox("History of Head Injury", ("No", "Yes"))
family_history = st.selectbox("Family History of Memory Loss or Dementia", ("No", "Yes"))
chronic_inflammation = st.selectbox("Chronic Conditions (e.g., Arthritis)", ("No", "Yes"))
socioeconomic_factors = st.selectbox("Social Support and Lifestyle Factors", 
                                     ("High", "Moderate", "Low"))

# Collect all inputs into a dictionary
input_data = {
    "age": age,
    "gender": gender,
    "smoking": smoking,
    "physical_activity": physical_activity,
    "head_trauma": head_trauma,
    "family_history": family_history,
    "chronic_inflammation": chronic_inflammation,
    "socioeconomic_factors": socioeconomic_factors
}

# Display input summary
st.write("### Summary of Inputs:")
st.json(input_data)

# Submit button for prediction
if st.button("Predict Alzheimer's Onset Risk"):
    if not api_key:
        st.error("Please enter your OpenAI API Key.")
    else:
        # Prepare the data for GPT-4 analysis
        prompt = f"""
        Given the following user details related to risk factors for Alzheimer's disease, provide an assessment of 
        potential onset risks in clear, easy-to-understand terms. Consider age, lifestyle, past injuries, family history, 
        and chronic conditions.
        
        User details:
        Age: {age}
        Gender: {gender}
        Smoking Status: {smoking}
        Physical Activity Level: {physical_activity}
        History of Head Injury: {head_trauma}
        Family History of Memory Loss or Dementia: {family_history}
        Chronic Conditions: {chronic_inflammation}
        Social Support and Lifestyle Factors: {socioeconomic_factors}
        
        Provide an assessment and potential predictive analysis on the Alzheimer's disease onset risk, suitable for a layperson's understanding.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # OpenAI API endpoint
        url = "https://api.openai.com/v1/chat/completions"

        # Data payload for OpenAI GPT-4 request
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        with st.spinner("Predicting Alzheimer's onset risk..."):
            try:
                # Make the POST request to OpenAI API
                response = requests.post(url, headers=headers, data=json.dumps(data))

                # Check for successful response
                if response.status_code == 200:
                    result = response.json()
                    # Get GPT-4 response text
                    output_text = result['choices'][0]['message']['content']
                    st.success("Prediction complete!")
                    st.write("### Alzheimer's Onset Risk Analysis:")
                    st.write(output_text)
                elif response.status_code == 401:
                    st.error("Authentication Error: The API key provided is incorrect or unauthorized.")
                elif response.status_code == 429:
                    st.error("Rate Limit Exceeded: You've made too many requests. Please try again later.")
                else:
                    error_message = response.json().get("error", {}).get("message", "An unknown error occurred.")
                    st.error(f"Error {response.status_code}: {error_message}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request Error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
