import os
import openai
import speech_recognition as sr
from llama_parse import LlamaParse
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
import json
import streamlit as st
import time
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import LangChainTracer
from langchain_openai import ChatOpenAI
from langsmith import Client

load_dotenv()

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

llm = OpenAI(
    model='gpt-3.5-turbo',
    temperature=0.0,
    max_tokens=500,
    api_key=OPENAI_API_KEY,
)

chat_model = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    callback_manager=callback_manager,
)

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

def parse_document(pdf_path):
    try:
        documents = parser.load_data(pdf_path)
        if not documents:
            raise ValueError("No documents were returned from LlamaParse.")
        document_text = documents[0].text
        if not document_text:
            raise ValueError("Failed to extract text content from the document.")
        return document_text
    except Exception as e:
        print(f"Error while parsing the file '{pdf_path}': {e}")
        return None

def find_similar_locations_with_gpt(parsed_text, pickup_location):
    message = f"""
    The following text contains various locations along with their coordinates. Your task is to find all locations that are similar to the pickup location '{pickup_location}' and return those locations along with their coordinates. Here is the text from the document:

    {parsed_text}

    Please provide the output in the following format:
    Location: <location_name>, Coordinates: [x, y, z]
    """

    try:
        response = chat_model.generate([[{"role": "user", "content": message}]])
        result = response.generations[0][0].text.strip()
        return result
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

def calculate_closest_pickup_location(similar_locations_text, dropoff_location_coords):
    message = f"""
    The following are potential pickup locations with their coordinates:

    {similar_locations_text}

    The drop-off location coordinates are: {dropoff_location_coords}

    Please determine which pickup location is closest to the drop-off location based on Euclidean distance, and provide the result in the following format:

    Closest Location: <location_name>, Coordinates: [x, y, z]
    """

    try:
        response = chat_model.generate([[{"role": "user", "content": message}]])
        result = response.generations[0][0].text.strip()
        return result
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

voice_to_text_tool = FunctionTool.from_defaults(fn=voice_to_text)
parse_command_for_locations_tool = FunctionTool.from_defaults(fn=parse_command_for_locations)
parse_document_tool = FunctionTool.from_defaults(fn=parse_document)
find_similar_locations_with_gpt_tool = FunctionTool.from_defaults(fn=find_similar_locations_with_gpt)
calculate_closest_pickup_location_tool = FunctionTool.from_defaults(fn=calculate_closest_pickup_location)  

agent1 = ReActAgent.from_tools(tools=[voice_to_text_tool], llm=llm, verbose=True)
agent3 = ReActAgent.from_tools(tools=[parse_document_tool], llm=llm, verbose=True)
agent4 = ReActAgent.from_tools(tools=[find_similar_locations_with_gpt_tool], llm=llm, verbose=True)
agent5 = ReActAgent.from_tools(tools=[calculate_closest_pickup_location_tool], llm=llm, verbose=True)  

def main():
    st.title("Delivery Command System")
    
    input_mode = st.radio("Choose input mode:", ("Voice", "Text"))
    
    if input_mode == "Voice":
        if st.button("Start Voice Input"):
            with st.spinner("Listening..."):
                command = agent1.chat("Use the voice_to_text tool to get the user's voice command.").response
            st.success(f"Voice Command: {command}")
    else:
        command = st.text_input("Please enter your command:")
    
    if command:
        st.write("\nProcessing the command...")
        
        with st.spinner("Extracting information..."):
            extracted_info = parse_command_for_locations(command)
        
        st.subheader("Extracted Information:")
        st.json(extracted_info)
        
        try:
            if extracted_info is None:
                raise ValueError("Failed to extract information from the command.")
            
            pickup_location = extracted_info.get('pickup_location')
            dropoff_location = extracted_info.get('dropoff_location')
            
            if pickup_location and dropoff_location:
                st.write(f"\nPickup Location: {pickup_location}")
                st.write(f"Dropoff Location: {dropoff_location}")
                
                pdf_path = "/Users/spartan/Desktop/pack-project/pack/waypoints_new.pdf"
                
                with st.spinner("Parsing document..."):
                    parse_result = agent3.chat(f"Parse the document at '{pdf_path}' using the parse_document tool.")
                    parsed_text = parse_result.response
                
                with st.spinner(f"Finding locations similar to '{pickup_location}'..."):
                    similar_locations_result = agent4.chat(f"Find locations similar to '{pickup_location}' using the find_similar_locations_with_gpt tool with the following parsed text:\n\n{parsed_text}")
                
                st.subheader(f"Locations similar to '{pickup_location}':")
                st.write(similar_locations_result.response)
                
                dropoff_location_coords = [6.19, -9.87, 1.92]
                
                with st.spinner("Calculating closest pickup location..."):
                    closest_location_result = agent5.chat(f"Use the calculate_closest_pickup_location tool to determine the closest pickup location from the following similar locations:\n\n{similar_locations_result.response}\n\nAnd the drop-off location coordinates: {dropoff_location_coords}")
                
                st.subheader(f"The closest pickup location to the drop-off point '{dropoff_location}' is:")
                st.success(closest_location_result.response)
            else:
                st.error("Could not extract pickup or dropoff location. Please try again with a different command.")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
        except Exception as e:
            st.error(f"Error processing the command: {str(e)}")

if __name__ == "__main__":
    main()