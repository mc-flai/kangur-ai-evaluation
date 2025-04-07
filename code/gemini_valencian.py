############################################################################# 
# This script is valid for evaluating the filtered questions of the
# Kangur test for different levels. It can be used for gemini-1.5-flash
# and gemini-2.0-flash-exp. An API key is needed (https://aistudio.google.com/apikey).
# Please, check the destination folders where responses will be saved as a txt
# file for each question.
#
#############################################################################

## Imports
import os
import pandas as pd
import re
import time
import google.generativeai as genai

## Read data 
df = pd.read_excel(r"kangur-ai-evaluation/dataset/dataset.xlsx")

## Choose your gemini version
version = "gemini-2.0-flash" # (or 'gemini-2.0-flash-lite')

## Destination folder (the LLM responses will be saved here as individual txt files)
dest = r"kangur-ai-evaluation/raw_responses/gemini-2.0-flash/valencian" # (or r"kangur-ai-evaluation/raw_responses/gemini-2.0-flash-lite")
if not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created {folder_path}")
## Input folder
input = r"kangur-ai-evaluation\dataset\Imagenes_Segmentadas\Valenciano"

## Configure gemini
GOOGLE_API_KEY = "your/api/key/here"
genai.configure(api_key = GOOGLE_API_KEY)
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
temperature = 0
top_p = 0
top_k = 1
generation_config = genai.GenerationConfig(temperature=temperature, top_p=top_p,top_k=top_k)
model = genai.GenerativeModel(version,
                              generation_config=generation_config,safety_settings=safety_settings)

## Auxiliar function
def extract_number_from_filename(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)  # Captures the last number before ".jpg"
    return int(match.group(1)) if match else None

## Loop: 1 API call per question
for file in os.listdir(input):
    # Skip file if it exists
    if os.path.exists(os.path.join(dest, f"{file}.txt")):
        print(f"Skipping file {file} as it already exists...")
        continue
      
    img_path = os.path.join(input,file)
    if img_path.endswith(".jpg"):
        EncodedImage = types.Part.from_bytes(
            data=pathlib.Path(img_path).read_bytes(),
            mime_type="image/jpeg"
        )
        print(f"Size in bytes of the image is : {EncodedImage.__sizeof__()/1024:2.2f}")

        # Question number
        extracted_int = extract_number_from_filename(file)

        # Prompt
        prompt = f"""Analitza la pregunta {extracted_int} mostrada en la imatge i tria la resposta correcta entre les opcions donades.
        **Instruccions**:
        Explica el teu raonament i proporciona la teua resposta final en aquest format específic, sense canvis:
            Raonament: Descriu el procés de pensament que t'ha portat a la resposta.
            Resposta: A), B), C), D) o E).
        """

        #print(prompt)

        try:
            response = client.models.generate_content(
            model=version,
            contents=[prompt,
                    EncodedImage])
            
            print(response.text)

            with open(os.path.join(dest, f"{file}.txt"), "w",encoding="utf-8") as f:
                f.write(f"{response.text}")

        except Exception as e:
            # if generating the response fails, wait 3 seconds to repeat
            time.sleep(3)
            continue
