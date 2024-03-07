import requests
from stix2 import MemoryStore

# https://arxiv.org/pdf/2306.14062.pdf
# https://pure.kfupm.edu.sa/en/publications/securebert-a-domain-specific-language-model-forcybersecurity
#
#

def get_data_from_branch(domain):
    # Get the ATT&CK STIX data from MITRE/CTI.
    # Domain should be 'enterprise-attack', 'mobile-attack' or 'ics-attack'.
    stix_json = requests.get(f"https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/{domain}/{domain}.json").json()
    return MemoryStore(stix_data=stix_json["objects"])

src = get_data_from_branch("enterprise-attack")

src.get("intrusion-set--f40eb8ce-2a74-4e56-89a1-227021410142")