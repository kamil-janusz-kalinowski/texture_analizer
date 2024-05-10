import json

# Load JSON file
path_report = r'models/models_data.json'
with open(path_report, 'r') as f:
    data = json.load(f)
    
print('End of script')
    