import random
from fpdf import FPDF

random.seed(42)

def generate_random_waypoint():
    x = round(random.uniform(-10, 10), 2)
    y = round(random.uniform(-10, 10), 2)
    theta = round(random.uniform(-3.14, 3.14), 2)  
    return [x, y, theta]

locations = [
    'Pharmacy1',
    'Pharmacy2',
    'Pharmacy3',
    'Starbucks1',
    'Starbucks2',
    'Library',
    'Gym',
    'Cafeteria1',
    'Cafeteria2',
    'Cafeteria3',
    'Conference Room A',
    'Reception',
    'Parking Lot',
    'Office 101',
    'Office 102',
    'Maintenance Room'
]

waypoints_data = []
for location in locations:
    waypoint = generate_random_waypoint()
    waypoints_data.append({
        'location': location,
        'waypoint': waypoint
    })

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="Waypoints Data", ln=True, align='C')
pdf.ln(10)

for data in waypoints_data:
    location = data['location']
    waypoint = data['waypoint']
    line = f"Location: {location}, Waypoint: {waypoint}"
    pdf.multi_cell(0, 10, line)
    pdf.ln(2)

pdf.output("waypoints_new.pdf")

print("Waypoints data has been generated and saved to 'waypoints.pdf'.")
