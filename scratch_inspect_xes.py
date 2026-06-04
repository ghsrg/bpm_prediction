import xml.etree.ElementTree as ET
import os

log_path = r"C:\Users\korsr\PycharmProjects\bpm_prediction\outputs\simulation\loan_v1_v5_complex_simulated.xes"

if not os.path.exists(log_path):
    print("Log not found at", log_path)
    exit()

print("Parsing XES log...")
tree = ET.parse(log_path)
root = tree.getroot()

# XES namespace
ns = {'xes': 'http://www.xes-standard.org/'}

version_activities = {} # version -> set of activities

for trace in root.findall('xes:trace', ns):
    # Find process version
    version = None
    for attr in trace.findall('xes:string', ns):
        if attr.attrib.get('key') == 'sim:process_version' or attr.attrib.get('key') == 'process_version':
            version = attr.attrib.get('value')
            break
            
    if not version:
        # Check trace attributes
        for attr in trace.findall('xes:string', ns):
            if 'version' in attr.attrib.get('key', ''):
                version = attr.attrib.get('value')
                break
                
    if not version:
        version = 'unknown'
        
    if version not in version_activities:
        version_activities[version] = set()
        
    for event in trace.findall('xes:event', ns):
        activity = None
        for attr in event.findall('xes:string', ns):
            if attr.attrib.get('key') == 'concept:name':
                activity = attr.attrib.get('value')
                break
        if activity:
            version_activities[version].add(activity)

print("\n=== Activities per Version ===")
for version, acts in sorted(version_activities.items()):
    print(f"Version {version}: {len(acts)} unique activities")
    
# Let's see what activities are in v3, v4, v5 but not in v1 and v2
seen_v1_v2 = version_activities.get('v1', set()).union(version_activities.get('v2', set()))
print(f"\nTotal unique activities in v1 & v2 (seen during training): {len(seen_v1_v2)}")

for v in ['v3', 'v4', 'v5']:
    v_acts = version_activities.get(v, set())
    new_acts = v_acts - seen_v1_v2
    print(f"Version {v} new activities: {len(new_acts)}")
    if new_acts:
        print(f"  New: {sorted(list(new_acts))}")
