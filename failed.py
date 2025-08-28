import json

# open all_failed_instances.json

with open('all_failed_instances.json', 'r') as f:
    data = json.load(f)
    print(len(data))