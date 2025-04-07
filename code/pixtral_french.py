############################################################################# 
# This script is valid for evaluating the filtered questions of the
# Kangur test for different levels. It can be used for Pixtral
# and Pixtral Large. An API key is needed (https://console.mistral.ai/api-keys).
# Please, check the destination folders where responses will be saved as a txt
# file for each question.
#
#############################################################################

## Imports
import os
import re
import base64
from io import BytesIO
from PIL import Image
from mistralai import Mistral
import time

## Model
API_KEY = "your/api/key"
client = Mistral(api_key=API_KEY)
version = "pixtral-12b" # or "pixtral-large-latest"

## Destination folder (the LLM responses will be saved here as individual txt files)
dest = r"kangur-ai-evaluation/raw_responses/pixtral-12b/french" # (or r"kangur-ai-evaluation/raw_responses/pixtral-large")
if not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created {folder_path}")
## Input folder
input = r"kangur-ai-evaluation\dataset\Imagenes_Segmentadas\Frances"

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
        prompt = f"""Analyse la question {extracted_int} montrée dans l'image et choisis la bonne réponse parmi les options proposées.
        **Instructions** :
        Explique ton raisonnement et fournis ta réponse finale dans ce format, sans aucune modification:
            Raisonnement : Décris le processus de réflexion qui t'a conduit à ta réponse.
            Réponse : A), B), C), D) ou E).
        """

        #print(prompt)

        try:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{EncodedImage}"}}
                    ]
                }
            ]

            # Perform inference
            response = client.chat.complete(
                model=version,
                messages=messages
            )

            print(response.choices[0].message.content)

            with open(os.path.join(dest, f"{file}.txt"), "w",encoding="utf-8") as f:
                f.write(f"{response.choices[0].message.content}")
        

        except Exception as e:
            time.sleep(10)
            print(e)
