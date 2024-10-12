from src.voice_interface.speech_to_text import convert_speech_to_text
from src.command_processing.command_parser import parse_command
from src.data_extraction.coordinate_extractor import extract_coordinates
from src.agent_system.multi_agent import MultiAgentSystem
from src.data_connectors.geo_connector import GeoConnector
from src.simulator.gazebo_interface import GazeboSimulator

def main():
    # Step 1: Convert voice command to text
    voice_command = convert_speech_to_text()

    # Step 2: Parse the command
    parsed_command = parse_command(voice_command)

    # Step 3: Extract coordinates
    pickup_coords, dropoff_coords = extract_coordinates(parsed_command)

    # Step 4: Use multi-agent system to process the request
    agent_system = MultiAgentSystem()
    processed_data = agent_system.process(pickup_coords, dropoff_coords)

    # Step 5: Use geo connector to get additional data if needed
    geo_connector = GeoConnector()
    additional_data = geo_connector.fetch_data(processed_data)

    # Step 6: Prepare data for simulation
    simulation_data = {
        "car": "car1",
        "pickup": pickup_coords,
        "dropoff": dropoff_coords,
        "additional_data": additional_data
    }

    # Step 7: Run simulation
    # simulator = GazeboSimulator()
    # simulator.run(simulation_data)

if __name__ == "__main__":
    main()