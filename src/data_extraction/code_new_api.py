import os
import openai
import speech_recognition as sr
from llama_parse import LlamaParse
from dotenv import load_dotenv
import json
import streamlit as st
import requests  # For sending payload to ROS API
import time
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import LangChainTracer
from langchain_openai import ChatOpenAI
from langsmith import Client
import streamlit.components.v1 as components

load_dotenv()

# API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

if not all([OPENAI_API_KEY, LLAMA_CLOUD_API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT]):
    raise ValueError("One or more required API keys or project name not found. Please ensure they're set correctly in the .env file.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY)

langsmith_client = Client(api_key=LANGCHAIN_API_KEY)
tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT)
callback_manager = CallbackManager([tracer])

chat_model = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    callback_manager=callback_manager,
)

# Helper functions
def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please say the command:")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"Voice Command: {text}")
            return text
        except Exception as e:
            st.error(f"Error in voice recognition: {str(e)}")
            return None

def parse_command_for_locations(command: str) -> dict:
    prompt = [
        {"role": "system", "content": "Extract the action, item, pickup location, and drop-off location from the command. Output ONLY the result in JSON format with keys 'action', 'item', 'pickup_location', and 'dropoff_location'. Do not include any additional text."},
        {"role": "user", "content": f'Command: "{command}"'},
    ]
    try:
        response = chat_model.generate([prompt])
        result = response.generations[0][0].text.strip()
        return json.loads(result)
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

def send_coordinates_to_ros(pickup_location_coords, dropoff_location_coords):
    ros_api_url = "http://ros_api_endpoint/move_robot"  # Replace with your actual ROS API endpoint
    payload = {
        "pickup_location": pickup_location_coords,
        "dropoff_location": dropoff_location_coords
    }
    try:
        response = requests.post(ros_api_url, json=payload)
        if response.status_code == 200:
            st.success("Successfully sent data to ROS")
        else:
            st.error(f"Failed to send data to ROS, status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error sending data to ROS: {e}")

def main():
    st.title("Autonomous Cart Command System")

    # Input mode selection (Voice/Text)
    input_mode = st.radio("Choose input mode:", ("Voice", "Text"))

    command = None  # Initialize command

    # Get command input based on selected mode
    if input_mode == "Voice":
        if st.button("Start Voice Input"):
            with st.spinner("Listening..."):
                command = voice_to_text()
            if command:
                st.success(f"Voice Command: {command}")
    else:
        command = st.text_input("Please enter your command:")

    if command:
        st.write("\nProcessing the command...")

        with st.spinner("Extracting pickup and dropoff locations..."):
            extracted_info = parse_command_for_locations(command)

        st.subheader("Extracted Information:")
        st.json(extracted_info)

        if extracted_info is None:
            st.warning("Could not extract locations. Please try again.")
            return

        # Get pickup and dropoff locations
        pickup_location = extracted_info.get('pickup_location')
        dropoff_location = extracted_info.get('dropoff_location')

        if pickup_location and dropoff_location:
            st.write(f"Pickup Location: {pickup_location}")
            st.write(f"Dropoff Location: {dropoff_location}")

            # Fake coordinates for demonstration purposes (replace with real logic)
            pickup_location_coords = [5.0, -3.2, 0.0]  # Example coordinates for pickup
            dropoff_location_coords = [10.0, 2.1, 0.0]  # Example coordinates for dropoff

            st.write("\nSending coordinates to ROS API...")

            with st.spinner("Sending data..."):
                send_coordinates_to_ros(pickup_location_coords, dropoff_location_coords)

            st.success("Operation completed successfully.")
        else:
            st.warning("Failed to extract pickup or dropoff location. Please check the command.")

if __name__ == "__main__":
    main()