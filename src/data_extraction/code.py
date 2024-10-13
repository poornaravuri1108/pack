import os
import openai
import speech_recognition as sr
from llama_parse import LlamaParse
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
import json

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')

if not OPENAI_API_KEY or not LLAMA_CLOUD_API_KEY:
    raise ValueError("API keys not found. Please ensure they're set correctly in the .env file.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY)

llm = OpenAI(
    model='gpt-3.5-turbo',
    temperature=0.0,
    max_tokens=500,
    api_key=OPENAI_API_KEY
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
    """
    Parses a delivery command to extract the action, item, pickup location, and drop-off location.

    Args:
        command (str): The delivery command.

    Returns:
        dict: A dictionary containing the extracted information with keys 'action', 'item', 'pickup_location', and 'dropoff_location'.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "Extract the action, item, pickup location, and drop-off location from the command. "
                "Output the result in JSON format with keys 'action', 'item', 'pickup_location', and 'dropoff_location'."
            ),
        },
        {"role": "user", "content": f'Command: "{command}"'},
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=500,
            temperature=0,
        )
        result = response.choices[0].message.content.strip()
        data = json.loads(result)
        return data
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None


def parse_document(pdf_path):
    print(f"Started parsing the file: {pdf_path}")
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
    print(f"*********************Parsed text is {parsed_text}******************")
    message = f"""Find locations that are same as '{pickup_location}' in the following text and return them with their exact coordinates:

                {parsed_text}

                Return the results in the following format:
                Location: [Location Name], Coordinates: [x, y, z]
                Location: [Location Name], Coordinates: [x, y, z]
                ...

                Only include locations that are actually present in the provided text and use the exact coordinates given."""

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

voice_to_text_tool = FunctionTool.from_defaults(fn=voice_to_text)
parse_command_for_locations_tool = FunctionTool.from_defaults(fn=parse_command_for_locations)
parse_document_tool = FunctionTool.from_defaults(fn=parse_document)
find_similar_locations_with_gpt_tool = FunctionTool.from_defaults(fn=find_similar_locations_with_gpt)

agent = ReActAgent.from_tools(
    tools=[voice_to_text_tool, parse_command_for_locations_tool],
    llm=llm,
    verbose=True,
)

agent1 = ReActAgent.from_tools(
    tools = [parse_document_tool, find_similar_locations_with_gpt_tool],
    verbose=True,
    llm=llm
)

print("Welcome to the Delivery Command System")

while True:
    input_mode = input("Would you like to input your command via voice or text? (v/t): ").strip().lower()

    if input_mode == 'v':
        try:
            command_response = agent.chat("Use the voice_to_text tool to get the user's voice command.")
            command = command_response.response
            print(f"Voice Command: {command}")
        except Exception as e:
            print(f"Error in voice recognition: {str(e)}")
            continue
    else:
        command = input("Please enter your command: ")

    print("\nProcessing the command...")

    try:
        result = agent.chat(
            f"Process this delivery command: '{command}'. Use the parse_command_for_locations tool to extract the action, item, pickup location, and drop-off location. Output the extracted information in JSON format."
        )

        print("\nExtracted Information:")
        print(result.response)

        try:
            extracted_info = json.loads(result.response)
        except json.JSONDecodeError as e:
            print(f"Error parsing the agent's response as JSON: {e}")
            extracted_info = None

        if extracted_info:
            pickup_location = extracted_info.get('pickup_location')
            dropoff_location = extracted_info.get('dropoff_location')

            if pickup_location and dropoff_location:
                print(f"\nPickup Location: {pickup_location}")
                print(f"Dropoff Location: {dropoff_location}")

                pdf_path = "/Users/spartan/Desktop/pack-project/pack/waypoints.pdf"
                try:
                    parse_result = agent1.chat(f"Parse the document at '{pdf_path}' using the parse_document tool.")
                    print("\nDocument parsed successfully.")

                    similar_locations_result = agent1.chat(
                        f"Find locations similar to '{pickup_location}' using the find_similar_locations_with_gpt tool."
                    )
                    print(f"\nLocations similar to '{pickup_location}':")

                    # Process and print the similar locations with their coordinates
                    locations = similar_locations_result.response.split('\n')
                    for location in locations:
                        if location.strip():
                            print(location.strip())

                except Exception as e:
                    print(f"Error in document parsing or finding similar locations: {str(e)}")
            else:
                print("Could not extract pickup or dropoff location. Please try again with a different command.")
        else:
            print("Failed to extract information from the agent's response.")

    except Exception as e:
        print(f"Error processing the command: {str(e)}")

    another = input("\nWould you like to process another command? (y/n): ").strip().lower()
    if another != 'y':
        break

print("Thank you for using the Delivery Command System!")