import os
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
import json
import logging
from pinecone import Pinecone
from openai import OpenAI as OpenAIClient

load_dotenv()

logging.basicConfig(level=logging.INFO)

pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')
pc = Pinecone(api_key=pinecone_api_key)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

index_name = 'location-index'
index = pc.Index(name=index_name)

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' does not exist in Pinecone.")

llm = OpenAI(model='gpt-4o', 
             temperature=0.0)

vector_store = PineconeVectorStore(index_name=index_name)

service_context = ServiceContext.from_defaults(llm=OpenAI(temperature=0.7, model="gpt-3.5-turbo"))

vector_index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)

openai_client = OpenAIClient()

def get_waypoint(location_name):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[location_name]
    )
    query_embedding = response.data[0].embedding
    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    print(f"Results for query matching {results}")

    if results['matches']:
        metadata = results['matches'][0]['metadata']
        return [float(metadata['waypoint_x']), float(metadata['waypoint_y']), float(metadata['waypoint_z'])]
    else:
        return None

waypoint_tool = FunctionTool.from_defaults(fn=get_waypoint)

system_prompt = """
You are an assistant that processes user requests and generates a JSON output in the following format:

{
  "tasks": [
    {
      "action": "pickup",
      "item": "antibiotics",
      "location": {
        "name": "Pharmacy1",
      },
      "waypoint": [x1, y1, theta1]
    },
    {
      "action": "pickup",
      "item": "coffee",
      "location": {
        "name": "Starbucks",
      },
      "waypoint": [x2, y2, theta2],
      "preferences": {
        "type": "latte",
        "size": "large"
      }
    }
  ]
}

Your job is to analyze the user's request, use the get_waypoint function to retrieve waypoints for each location, and output the tasks in the JSON format above. If a waypoint is not found for a location, use [0, 0, 0] as the default.
"""
agent = ReActAgent.from_tools(
    tools=[waypoint_tool],
    llm=OpenAI(temperature=0.0, model="gpt-4o"),
    verbose=True,
    system_prompt=system_prompt
)

def process_user_request(user_input):
    response = agent.chat(user_input)

    try:
        json_response = json.loads(response.response)
        return json_response
    except json.JSONDecodeError:
        return {"error": "Failed to generate valid JSON"}

user_request = "Pick up a large latte from Starbucks."
result = process_user_request(user_request)
print(json.dumps(result, indent=2))

# def get_waypoint(location_name: str) -> dict:
#     response = openai_client.embeddings.create(
#         model="text-embedding-ada-002",
#         input=[location_name]
#     )
#     query_embedding = response.data[0].embedding
#     results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

#     if results['matches']:
#         metadata = results['matches'][0]['metadata']
#         return {
#             "name": metadata['location'],
#             "waypoint": [float(metadata['waypoint_x']), float(metadata['waypoint_y']), float(metadata['waypoint_z'])]
#         }
#     else:
#         return {"name": location_name, "waypoint": [0, 0, 0]}

# def process_user_request(user_input: str) -> dict:
#     tasks = user_input.split(" and then ")
#     json_output = {"tasks": []}

#     for task in tasks:
#         action = "pickup"
#         item = task.split("from")[0].strip().split()[-1]
#         location_name = task.split("from")[-1].strip().split()[0]

#         location_info = get_waypoint(location_name)

#         task_json = {
#             "action": action,
#             "item": item,
#             "location": {
#                 "name": location_info["name"]
#             },
#             "waypoint": location_info["waypoint"]
#         }

#         if "coffee" in item.lower():
#             preferences = {}
#             if "large" in task.lower():
#                 preferences["size"] = "large"
#             if "latte" in task.lower():
#                 preferences["type"] = "latte"
#             if preferences:
#                 task_json["preferences"] = preferences

#         json_output["tasks"].append(task_json)

#     return json_output

# if __name__ == "__main__":
#     user_request = "I need to pick up antibiotics from Pharmacy1 and then get a large latte from Starbucks."
#     result = process_user_request(user_request)
#     print(json.dumps(result, indent=2))