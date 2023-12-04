from pymongo import MongoClient
from bunnet import Document, Indexed, init_bunnet

class Incident(Document):
    number: str
    description: str
    group: str
    assigned_to: str

    class Settings:
        name = "incidents"

def init() -> None:
    client = MongoClient()
    init_bunnet(database=client.tickets, document_models=[Incident])

def main():
    init() # initalize the database

    incidents = [
        Incident(number="0001", description="User unable to login", group="Linux/Unix", assigned_to="James"),
        Incident(number="0002", description="Server non-responsive", group="Windows", assigned_to="Tom"),
        Incident(number="0003", description="Unable to enter time", group="HR", assigned_to="Nancy")
    ]
    Incident.insert_many(incidents)

if __name__ == "__main__":
    main()