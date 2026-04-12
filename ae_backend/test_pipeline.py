import requests
import time

url = "http://127.0.0.1:8085/api/ae/pipeline/prepare"
payload = {
    "region_name": "和平村",
    "satellite_sources": [
        "Sentinel-2", # Public GEE Source
        "D:/adk/data_agent/weights/raw_data/dummy_high_res_private.tif" # Private Local Source
    ],
    "bounding_box": [106.116, 29.601, 106.144, 29.644]
}

print("Sending request to Data Fusion Pipeline...")
try:
    response = requests.post(url, json=payload, timeout=10, proxies={"http": None, "https": None})
    print(f"Response Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
except Exception as e:
    print(f"Request failed: {e}")

print("Waiting for background task to process...")
time.sleep(3) # Wait a bit for the pipeline to start logging
