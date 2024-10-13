import os
import openai
import speech_recognition as sr
from llama_parse import LlamaParse
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
import json

# pip install llama-index==0.10.68
# pip install llama-agents==0.0.14
# pip install llama-index-vector-stores-pinecone==0.1.2

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')

if not OPENAI_API_KEY or not LLAMA_CLOUD_API_KEY:
    raise ValueError("API keys not found. Please ensure they're set correctly in the .env file.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY)

llm = OpenAI(
    model='gpt-4',
    temperature=0.0,
    max_tokens=500,
    api_key=OPENAI_API_KEY,
)

def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say the command:")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Voice Command: {text}")
            return text
        except Exception as e:
            print(f"Error in voice recognition: {str(e)}")
            return None

def parse_command_for_locations(command: str) -> dict:
    prompt = [
        {"role": "system", "content": "Extract the action, item, pickup location, and drop-off location from the command. Output ONLY the result in JSON format with keys 'action', 'item', 'pickup_location', and 'dropoff_location'. Do not include any additional text."},
        {"role": "user", "content": f'Command: "{command}"'},
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=prompt,
            max_tokens=500,
            temperature=0,
        )
        result = response.choices[0].message.content.strip()
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
            max_tokens=500,
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

def calculate_closest_pickup_location(similar_locations_text, dropoff_location_coords):
    """
    Instead of performing mathematical calculations, this function delegates the task to GPT-3.5-turbo.
    """
    message = f"""
    The following are potential pickup locations with their coordinates:

    {similar_locations_text}

    The drop-off location coordinates are: {dropoff_location_coords}

    Please determine which pickup location is closest to the drop-off location based on Euclidean distance, and provide the result in the following format:

    Closest Location: <location_name>, Coordinates: [x, y, z]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
            max_tokens=150,
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

# Define the tools
voice_to_text_tool = FunctionTool.from_defaults(fn=voice_to_text)
parse_command_for_locations_tool = FunctionTool.from_defaults(fn=parse_command_for_locations)
parse_document_tool = FunctionTool.from_defaults(fn=parse_document)
find_similar_locations_with_gpt_tool = FunctionTool.from_defaults(fn=find_similar_locations_with_gpt)
calculate_closest_pickup_location_tool = FunctionTool.from_defaults(fn=calculate_closest_pickup_location)  # Updated tool

# Initialize agents
agent1 = ReActAgent.from_tools(tools=[voice_to_text_tool], llm=llm, verbose=True)
agent3 = ReActAgent.from_tools(tools=[parse_document_tool], llm=llm, verbose=True)
agent4 = ReActAgent.from_tools(tools=[find_similar_locations_with_gpt_tool], llm=llm, verbose=True)
agent5 = ReActAgent.from_tools(tools=[calculate_closest_pickup_location_tool], llm=llm, verbose=True)  # New agent for calculation

print("Welcome to the Delivery Command System")

while True:
    input_mode = input("Would you like to input your command via voice or text? (v/t): ").strip().lower()

    if input_mode == 'v':
        command = agent1.chat("Use the voice_to_text tool to get the user's voice command.").response
        print(f"Voice Command: {command}")
    else:
        command = input("Please enter your command: ")

    print("\nProcessing the command...")

    # Call parse_command_for_locations directly
    extracted_info = parse_command_for_locations(command)

    print("\nExtracted Information:")
    print(extracted_info)

    try:
        if extracted_info is None:
            raise ValueError("Failed to extract information from the command.")

        pickup_location = extracted_info.get('pickup_location')
        dropoff_location = extracted_info.get('dropoff_location')

        if pickup_location and dropoff_location:
            print(f"\nPickup Location: {pickup_location}")
            print(f"Dropoff Location: {dropoff_location}")

            pdf_path = "waypoints_new.pdf"

            # Use parse_document via agent3
            parse_result = agent3.chat(f"Parse the document at '{pdf_path}' using the parse_document tool.")
            parsed_text = parse_result.response

            # Use find_similar_locations_with_gpt via agent4
            similar_locations_result = agent4.chat(f"Find locations similar to '{pickup_location}' using the find_similar_locations_with_gpt tool with the following parsed text:\n\n{parsed_text}")
            print(f"\nLocations similar to '{pickup_location}':")
            print(similar_locations_result.response)

            # Hard-coded drop-off location coordinates (you can adjust these values)
            dropoff_location_coords = [6.19, -9.87, 1.92]  # Example coordinates for 'Gym'

            # Use calculate_closest_pickup_location via agent5
            closest_location_result = agent5.chat(f"Use the calculate_closest_pickup_location tool to determine the closest pickup location from the following similar locations:\n\n{similar_locations_result.response}\n\nAnd the drop-off location coordinates: {dropoff_location_coords}")
            print(f"\nThe closest pickup location to the drop-off point '{dropoff_location}' is:")
            print(closest_location_result.response)
        else:
            print("Could not extract pickup or dropoff location. Please try again with a different command.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error processing the command: {str(e)}")

    another = input("\nWould you like to process another command? (y/n): ").strip().lower()
    if another != 'y':
        break

print("Thank you for using the Delivery Command System!")
