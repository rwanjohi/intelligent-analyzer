"""
Generate synthetic training data for claim type classification.
Creates a CSV with claim descriptions, claim types, and locations.
"""

import pandas as pd
import random
import os

random.seed(42)

# Claim types and templates
CLAIM_TEMPLATES = {
    "Auto Accident": [
        "Vehicle collision occurred at intersection in {city}, {state}. Front bumper damaged and airbags deployed.",
        "Rear-ended at traffic light in {city}, {state}. Significant damage to rear of vehicle, neck injury reported.",
        "Two-car accident on highway near {city}, {state}. Both vehicles totaled, minor injuries.",
        "Side collision in parking lot of mall in {city}, {state}. Driver door dented severely.",
        "Hit and run incident in {city}, {state}. Driver fled scene after damaging passenger side.",
        "Single vehicle accident on icy road in {city}, {state}. Vehicle slid into guardrail.",
        "Multi-car pileup on freeway in {city}, {state} during heavy fog. Multiple injuries reported.",
        "Collision with deer on rural road outside {city}, {state}. Front end significantly damaged.",
        "T-bone accident at four-way stop in {city}, {state}. Driver ran stop sign causing collision.",
        "Fender bender at gas station in {city}, {state} when other vehicle reversed without looking.",
    ],
    "Property Damage": [
        "Tree fell on roof during storm in {city}, {state}. Significant structural damage to attic.",
        "Burst pipe in basement of home in {city}, {state}. Water damage to flooring and walls.",
        "Hail damage to roof and siding of property in {city}, {state}. Multiple broken windows.",
        "Fire damage to kitchen of residence in {city}, {state}. Smoke damage throughout home.",
        "Vandalism to property in {city}, {state}. Graffiti on exterior walls and broken windows.",
        "Wind damage tore shingles off roof in {city}, {state}. Interior leaking after rain.",
        "Flooding from heavy rain damaged finished basement in {city}, {state}.",
        "Tornado damage to home in {city}, {state}. Detached garage destroyed completely.",
        "Lightning strike caused electrical fire at residence in {city}, {state}.",
        "Frozen pipes burst causing extensive water damage to home in {city}, {state}.",
    ],
    "Medical": [
        "Slip and fall at grocery store in {city}, {state}. Broken hip requiring surgery and rehabilitation.",
        "Workplace injury at warehouse in {city}, {state}. Back strain from heavy lifting incident.",
        "Surgery complications following knee replacement at hospital in {city}, {state}.",
        "Emergency room visit for chest pains in {city}, {state}. Cardiac event requiring stent.",
        "Broken arm from playground accident in {city}, {state}. Child required cast and follow-up.",
        "Diagnostic imaging and specialist consultation in {city}, {state} for chronic back pain.",
        "Inpatient hospitalization for pneumonia treatment in {city}, {state} for five days.",
        "Physical therapy sessions in {city}, {state} following sports-related shoulder injury.",
        "Outpatient surgery for gallbladder removal at clinic in {city}, {state}.",
        "Maternity care and delivery at hospital in {city}, {state}. Cesarean section required.",
    ],
    "Theft": [
        "Burglary at residence in {city}, {state}. Electronics, jewelry, and cash stolen overnight.",
        "Vehicle stolen from driveway in {city}, {state}. Recovered three days later with damage.",
        "Package theft from porch in {city}, {state}. Multiple deliveries taken over past month.",
        "Bicycle stolen from rack outside business in {city}, {state}. Lock cut and bike removed.",
        "Identity theft and fraudulent charges affecting resident of {city}, {state}.",
        "Smash and grab at parked car in {city}, {state}. Laptop and bag taken from back seat.",
        "Shoplifting incident at retail store in {city}, {state}. Significant inventory loss.",
        "Catalytic converter stolen from vehicle parked in {city}, {state}.",
        "Office break-in in {city}, {state}. Computers and equipment stolen overnight.",
        "Mail theft from mailbox cluster in {city}, {state}. Checks and personal info compromised.",
    ],
    "Natural Disaster": [
        "Hurricane caused extensive damage to coastal property in {city}, {state}. Roof torn off.",
        "Wildfire destroyed structures and vehicles at property in {city}, {state}.",
        "Earthquake caused foundation damage to home in {city}, {state}. Cracks throughout walls.",
        "Severe flooding from river overflow damaged property in {city}, {state}.",
        "Mudslide following heavy rain damaged home foundation in {city}, {state}.",
        "Tornado completely destroyed barn and damaged house in {city}, {state}.",
        "Blizzard caused roof collapse from snow accumulation in {city}, {state}.",
        "Tropical storm flooding ruined first floor of business in {city}, {state}.",
        "Hail storm shattered greenhouse and damaged crops in {city}, {state}.",
        "Severe ice storm brought down power lines and trees on property in {city}, {state}.",
    ],
    "Liability": [
        "Customer slipped on wet floor at restaurant in {city}, {state}. Suing for medical bills.",
        "Dog bite incident in {city}, {state}. Neighbor required stitches and treatment.",
        "Faulty product injured user in {city}, {state}. Product liability claim filed.",
        "Construction worker injured at site in {city}, {state}. Premises liability claim.",
        "Pool accident at rental property in {city}, {state}. Guest required emergency care.",
        "Trip and fall on broken sidewalk outside business in {city}, {state}.",
        "Food poisoning at catered event in {city}, {state}. Multiple guests hospitalized.",
        "Customer injury from falling merchandise at store in {city}, {state}.",
        "Tenant injury claim from defective stairs at apartment in {city}, {state}.",
        "Negligent security claim following assault on premises in {city}, {state}.",
    ],
    "Workers Compensation": [
        "Construction worker fell from scaffolding at site in {city}, {state}. Multiple fractures.",
        "Office worker developed carpal tunnel syndrome at company in {city}, {state}.",
        "Factory worker injured by machinery at plant in {city}, {state}. Hand laceration.",
        "Nurse injured back lifting patient at hospital in {city}, {state}.",
        "Delivery driver injured in fall while making delivery in {city}, {state}.",
        "Restaurant employee burned by hot oil at kitchen in {city}, {state}.",
        "Warehouse worker struck by forklift at facility in {city}, {state}.",
        "Teacher injured breaking up altercation at school in {city}, {state}.",
        "Mechanic injured by falling vehicle at shop in {city}, {state}.",
        "Retail employee injured during armed robbery at store in {city}, {state}.",
    ],
}

# State + city pairs for realistic location data
LOCATIONS = [
    ("New York", "NY"), ("Buffalo", "NY"), ("Albany", "NY"), ("Rochester", "NY"),
    ("Los Angeles", "CA"), ("San Francisco", "CA"), ("San Diego", "CA"), ("Sacramento", "CA"),
    ("Chicago", "IL"), ("Springfield", "IL"), ("Peoria", "IL"),
    ("Houston", "TX"), ("Dallas", "TX"), ("Austin", "TX"), ("San Antonio", "TX"),
    ("Phoenix", "AZ"), ("Tucson", "AZ"), ("Mesa", "AZ"),
    ("Philadelphia", "PA"), ("Pittsburgh", "PA"), ("Allentown", "PA"),
    ("Miami", "FL"), ("Orlando", "FL"), ("Tampa", "FL"), ("Jacksonville", "FL"),
    ("Atlanta", "GA"), ("Savannah", "GA"), ("Augusta", "GA"),
    ("Boston", "MA"), ("Worcester", "MA"), ("Springfield", "MA"),
    ("Seattle", "WA"), ("Spokane", "WA"), ("Tacoma", "WA"),
    ("Denver", "CO"), ("Boulder", "CO"), ("Colorado Springs", "CO"),
    ("Detroit", "MI"), ("Grand Rapids", "MI"), ("Lansing", "MI"),
    ("Minneapolis", "MN"), ("Saint Paul", "MN"), ("Duluth", "MN"),
    ("Portland", "OR"), ("Eugene", "OR"), ("Salem", "OR"),
    ("Las Vegas", "NV"), ("Reno", "NV"),
    ("Nashville", "TN"), ("Memphis", "TN"), ("Knoxville", "TN"),
    ("Charlotte", "NC"), ("Raleigh", "NC"), ("Asheville", "NC"),
    ("Columbus", "OH"), ("Cleveland", "OH"), ("Cincinnati", "OH"),
    ("Indianapolis", "IN"), ("Fort Wayne", "IN"),
    ("Kansas City", "MO"), ("Saint Louis", "MO"), ("Springfield", "MO"),
    ("Milwaukee", "WI"), ("Madison", "WI"), ("Green Bay", "WI"),
    ("New Orleans", "LA"), ("Baton Rouge", "LA"),
    ("Oklahoma City", "OK"), ("Tulsa", "OK"),
    ("Salt Lake City", "UT"), ("Provo", "UT"),
    ("Newark", "NJ"), ("Jersey City", "NJ"), ("Trenton", "NJ"),
    ("Baltimore", "MD"), ("Annapolis", "MD"),
    ("Richmond", "VA"), ("Norfolk", "VA"), ("Arlington", "VA"),
]

def generate_data(samples_per_class=80):
    rows = []
    claim_id = 10000
    for claim_type, templates in CLAIM_TEMPLATES.items():
        for _ in range(samples_per_class):
            template = random.choice(templates)
            city, state = random.choice(LOCATIONS)
            description = template.format(city=city, state=state)

            # Add small noise/variations to descriptions
            if random.random() < 0.3:
                description += f" Claim filed on {random.randint(1,28):02d}/{random.randint(1,12):02d}/2024."
            if random.random() < 0.2:
                description += f" Estimated damage ${random.randint(500, 50000):,}."
            if random.random() < 0.15:
                description += " Witness statements collected at scene."

            rows.append({
                "claim_id": f"CLM-{claim_id}",
                "description": description,
                "claim_type": claim_type,
                "claim_amount": random.randint(500, 100000),
                "date_filed": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            })
            claim_id += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    df = generate_data(samples_per_class=80)
    out_path = os.path.join(out_dir, "training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["claim_type"].value_counts())

    # Also generate a sample inference file for the user to test.
    # Keep claim_type in this file so the UI can show actual vs predicted
    # plus a match Y/N column.
    sample_df = generate_data(samples_per_class=8)
    sample_path = os.path.join(out_dir, "sample_claims.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Sample test file -> {sample_path}")
