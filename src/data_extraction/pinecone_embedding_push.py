import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import PyPDF2
import re
from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')
pc = Pinecone(api_key=pinecone_api_key)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

index_name = 'location-index'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='gcp', region='us-west1')
    )
index = pc.Index(index_name)

def read_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
    return text_content

def parse_locations(text):
    pattern = r"Location: (.*?), Waypoint: \[(.*?),(.*?),(.*?)\]"
    matches = re.findall(pattern, text)
    parsed_data = []
    for match in matches:
        location = match[0]
        waypoint = [str(float(match[1])), str(float(match[2])), str(float(match[3]))]
        parsed_data.append({
            'location': location,
            'waypoint': waypoint
        })
    return parsed_data

def create_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def process_locations(pdf_path):
    text_content = read_pdf(pdf_path)
    parsed_data = parse_locations(text_content)

    for data in parsed_data:
        location = data['location']
        waypoint = data['waypoint']
        record_id = location.replace(' ', '_').lower()

        existing_record = index.fetch(ids=[record_id])
        if existing_record and record_id in existing_record['vectors']:
            print(f"Skipping existing record: {record_id}")
            continue

        embedding = create_embedding(location)

        metadata = {
            'location': location,
            'waypoint_x': waypoint[0],
            'waypoint_y': waypoint[1],
            'waypoint_z': waypoint[2]
        }

        index.upsert([(record_id, embedding, metadata)])
        print(f"Upserted record: {record_id}")

if __name__ == "__main__":
    pdf_file_path = '/Users/spartan/Desktop/pack-project/pack/waypoints.pdf' 
    process_locations(pdf_file_path)