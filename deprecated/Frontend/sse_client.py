import ast
import base64
import io

import requests
from PIL import Image

api_key = "your-secret-api-key"

headers = {"key": api_key, "Content-Type": "application/json"}


def parse_sse(response):
    """Parse Server-Sent Events from a response."""
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                yield line[6:]  # Remove 'data: ' prefix


def process_and_display_stream(url, input_data):
    try:
        # Send the initial POST request with JSON data
        response = requests.post(url, json=input_data, headers=headers, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        for data in parse_sse(response):
            try:
                event = ast.literal_eval(data)
                if type(event) is dict:
                    if "image" in event.keys():
                        image_data = event["image"]
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        image.show()
                        print("Image received and displayed.")
                    else:
                        print(f"Received data: {data}")
            except:
                print(f"Received non-dict data: {data}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    server_url = "http://localhost/photonapi"  # Adjust this to your server's address

    input_data = {"text": "a 2x2 mmi", "parameter1": 3, "parameter2": True}

    process_and_display_stream(server_url, input_data)
