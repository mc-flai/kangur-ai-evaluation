############################################################################# 
# This script is valid for evaluating the filtered questions of the
# Kangur test for different levels. It can be used for LLAMA 11B
# and LLAMA 90B (Vision). An API key is needed (https://build.nvidia.com/models).
# Please, check the destination folders where responses will be saved as a txt
# file for each question.
#
#############################################################################

## Imports
import os
import re
import base64
from io import BytesIO
import time
import requests
import json

## Read data 
df = pd.read_excel(r"kangur-ai-evaluation/dataset/dataset.xlsx")

## Model
invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

headers = {
  "Authorization": "your nvidia api key here",
  "Accept": "text/event-stream" if stream else "application/json"
}

## Destination folder (the LLM responses will be saved here as individual txt files)
dest = r"kangur-ai-evaluation/raw_responses/llama-11b/spanish" # (or r"kangur-ai-evaluation/raw_responses/llama-90b")
if not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created {folder_path}")
## Input folder
input = r"kangur-ai-evaluation\dataset\Imagenes_Segmentadas\Castellano"

## Auxiliar function
def extract_number_from_filename(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)  # Captures the last number before ".jpg"
    return int(match.group(1)) if match else None
  
# Function to read the image file and convert it to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

for file in os.listdir(input):
    if os.path.exists(os.path.join(dest, f"{file}.txt")):
        print(f"Skipping file {file} as it already exists...")
        continue
    img_path = os.path.join(input,file)
    if img_path.endswith(".jpg"):

        EncodedImage = encode_image_to_base64(img_path)

        print(f"Size in bytes of the image is : {EncodedImage.__sizeof__()/1024:2.2f}")

        # Question number
        extracted_int = extract_number_from_filename(file)

        # Prompt
        prompt = f"""Analiza la pregunta {extracted_int} mostrada en la imagen y elige la respuesta correcta entre las opciones dadas.
        **Instrucciones**:
        Explica tu razonamiento y proporciona tu respuesta final en este formato específico, sin cambios:
            Razonamiento: Describe el proceso de pensamiento que te llevó a la respuesta.
            Respuesta: A), B), C), D) o E).
        """

        #print(prompt)

        try:

            payload = {
            "model": 'meta/llama-3.2-11b-vision-instruct',
            "messages": [
                {
                "role": "user",
                "content": f'{prompt} <img src="data:image/png;base64,{EncodedImage}" />'
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
            "top_p": 0,
            "stream": stream
            }

            response = requests.post(invoke_url, headers=headers, json=payload)


            with open(os.path.join(dest, f"{file}.txt"), "w", encoding="utf-8") as f:
                if stream:
                    # Process each chunk in the stream
                    for line in response.iter_lines():
                        if line:
                            # Decode the line and find the actual text in 'content'
                            line_str = line.decode("utf-8")
                            
                            # Assuming the content is stored within a key like "content", 
                            # extract it from the chunk (if it's in JSON format)
                            if "data: " in line_str:  # The line contains data in chunks
                                # Try to extract the content field from the JSON-like string
                                try:
                                    # Strip off "data: " and load it as a JSON string
                                    json_content = line_str[len("data: "):]
                                    chunk_data = json.loads(json_content)
                                    
                                    # Get the text content
                                    content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    
                                    # Write content to the file
                                    if content:
                                        f.write(content)  # Add the content to the file
                                except Exception as e:
                                    print(f"Error processing chunk: {e}")
                else:
                    # In case it's a non-streaming response, you can just write the JSON to a file
                    f.write(response.json())
        

        except Exception as e:
            time.sleep(10)
            print(e)
