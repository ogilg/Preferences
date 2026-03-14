import json

with open("experiments/token_level_probes/data/harm.json") as f:
    data = json.load(f)

for e in data:
    if e["turn"] == "user" and e["condition"] == "harmful":
        idx = e["id"].split("_")[1]
        print(f"--- {e['id']} ---")
        print(f"  prompt: {e['messages'][0]['content']}")
        print(f"  span:   {e['critical_span']}")
        # Find matching benign
        benign = [x for x in data if x["id"] == f"harm_{idx}_benign_user"][0]
        print(f"  benign: {benign['critical_span']}")
        nonsense = [x for x in data if x["id"] == f"harm_{idx}_nonsense_user"][0]
        print(f"  nonsns: {nonsense['critical_span']}")
        print()
