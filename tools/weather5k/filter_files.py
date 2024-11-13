import json
with open('/home/bingxing2/ailab/group/ai4earth/data/weather-5k/WEATHER-5K/meta_info.json', 'r', encoding='utf-8') as file:
    meta_data = json.load(file)


def filter_files_by_region(data, min_longitude, max_longitude, min_latitude, max_latitude):
    filtered_data = {}
    for filename, info in data.items():
        converted_longitude = convert_longitude(info["longitude"])
        if (min_longitude <= converted_longitude <= max_longitude and
            min_latitude <= info["latitude"] <= max_latitude):
            filtered_data[filename] = info
    return filtered_data

def convert_longitude(longitude):
    """Convert longitude from -180~180 to 0~360."""
    return longitude + 360 if longitude < 0 else longitude

# Example: Define the region
# min_longitude = 239.
# max_longitude = 286.
# min_latitude = 25.
# max_latitude = 47

min_longitude = 276.
max_longitude = 284.
min_latitude = 36.
max_latitude = 44

# Get filtered files
filtered_files = filter_files_by_region(meta_data, min_longitude, max_longitude, min_latitude, max_latitude)
print(filtered_files)
# Save to JSON file
with open('meta_info_weather5k_hrrr_r1.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_files, f, ensure_ascii=False, indent=4)

print("Filtered data saved to 'filtered_files.json'")