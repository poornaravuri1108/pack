import dotenv
dotenv.load_dotenv()  # our .env defines OPENAI_API_KEY
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent import FnAgentWorker
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    PipelineOrchestrator,
    ServiceComponent,
)
from llama_agents.launchers import LocalLauncher
from llama_index.llms.openai import OpenAI
import logging

from src.voice_interface.speech_to_text import convert_speech_to_text
from src.command_processing.command_parser import parse_command
from src.data_extraction.coordinate_extractor import extract_coordinates
from src.data_connectors.geo_connector import GeoConnector
from src.simulator.gazebo_interface import GazeboSimulator

# Set up logging
logging.getLogger("llama_agents").setLevel(logging.INFO)

# Define agent functions
def voice_recognition_fn(state):
    audio_input = state["__task__"].input
    text = convert_speech_to_text(audio_input)
    state["__output__"] = text
    return state, True

def command_parsing_fn(state):
    text_command = state["__task__"].input
    parsed_command = parse_command(text_command)
    state["__output__"] = parsed_command
    return state, True

def coordinate_extraction_fn(state):
    parsed_command = state["__task__"].input
    coordinates = extract_coordinates(parsed_command)
    state["__output__"] = coordinates
    return state, True

def route_planning_fn(state):
    coordinates = state["__task__"].input
    geo_connector = GeoConnector()
    route = geo_connector.plan_route(coordinates)
    state["__output__"] = route
    return state, True

def simulation_fn(state):
    route = state["__task__"].input
    simulator = GazeboSimulator()
    result = simulator.run(route)
    state["__output__"] = result
    return state, True

# Create agent workers
voice_agent = FnAgentWorker(fn=voice_recognition_fn).as_agent()
parsing_agent = FnAgentWorker(fn=command_parsing_fn).as_agent()
coordinate_agent = FnAgentWorker(fn=coordinate_extraction_fn).as_agent()
route_agent = FnAgentWorker(fn=route_planning_fn).as_agent()
simulation_agent = FnAgentWorker(fn=simulation_fn).as_agent()

# Set up the multi-agent system
message_queue = SimpleMessageQueue()

# Create agent services
voice_service = AgentService(
    agent=voice_agent,
    message_queue=message_queue,
    description="Voice recognition service",
    service_name="voice_recognition",
)

parsing_service = AgentService(
    agent=parsing_agent,
    message_queue=message_queue,
    description="Command parsing service",
    service_name="command_parsing",
)

coordinate_service = AgentService(
    agent=coordinate_agent,
    message_queue=message_queue,
    description="Coordinate extraction service",
    service_name="coordinate_extraction",
)

route_service = AgentService(
    agent=route_agent,
    message_queue=message_queue,
    description="Route planning service",
    service_name="route_planning",
)

simulation_service = AgentService(
    agent=simulation_agent,
    message_queue=message_queue,
    description="Simulation service",
    service_name="simulation",
)

# Create the pipeline
pipeline = QueryPipeline(chain=[
    ServiceComponent.from_service_definition(voice_service),
    ServiceComponent.from_service_definition(parsing_service),
    ServiceComponent.from_service_definition(coordinate_service),
    ServiceComponent.from_service_definition(route_service),
    ServiceComponent.from_service_definition(simulation_service),
])
orchestrator = PipelineOrchestrator(pipeline)

control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=orchestrator,
)

# Set up the launcher
launcher = LocalLauncher(
    [voice_service, parsing_service, coordinate_service, route_service, simulation_service],
    control_plane,
    message_queue,
)

def main():
    # Assuming we have an audio input, replace this with actual audio input method
    audio_input = "Sample audio input"
    
    # Run the agentic workflow
    result = launcher.launch_single(audio_input)
    print(result)

if __name__ == "__main__":
    main()