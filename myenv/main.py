import json
import os
import random
import re
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Focus: Node.js + React hallucinations
languages = ["JavaScript"]
models = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",  # Keep stable models
    # "qwen/qwen3-32b",  # commented out (unstable JSON)
]

hallucination_types = [
    "API_Library_Misuse",
    "Security_Critical_Hallucination",
    "Integration_Deployment_Hallucination",
    "Correct_Code"
]

COLUMNS = [
    "language",
    "hallucination_type",
    "code_snippet",
    "description",
    "hallucination_details",
    "model",
]

def get_prompt_with_hallucination_type(language, hallucination_type):
    return f"""
You are an expert full-stack {language} developer.

Generate a realistic {language} code snippet (20–80 lines) that represents **one** of the following categories:

1. **API/Library Misuse** — incorrect or invented use of an API, library, or function.
   Example: calling non-existent Express.js functions, using a fake React hook, or misusing Mongoose methods.

2. **Security-Critical Hallucination** — insecure or fabricated logic related to authentication, encryption, or security.
   Example: comparing passwords as plain strings, using made-up crypto APIs, or unsafe JWT handling.

3. **Integration/Deployment Hallucination** — invalid configurations, imaginary environment variables, or incorrect Docker/GitHub CI setups.
   Example: fake config keys in package.json, invalid YAML keys, non-existent Docker base images.

4. **Correct Code** — realistic, production-quality Node.js + React snippet that is well-structured and secure.

---
Requirements:
- Alternate between frontend (React) and backend (Node.js + Express) logic across generations.
- The code must look like something from a real project (not toy examples).
- If it’s a hallucination, make it look plausible but factually wrong.
- If it’s correct, make it actually valid and secure.
- Output must be valid JSON. Properly escape all quotes and newlines inside code.
- Do NOT explain or comment outside JSON.

Respond in **strict JSON**:
{{
  "language": "{language}",
  "hallucination_type": "{hallucination_type}",
  "code_snippet": "<full {language} code here, escaped properly>",
  "description": "<brief summary of what this code is supposed to do>",
  "hallucination_details": "<if hallucinated: describe the incorrect or invented part; if correct: 'none'>"
}}
"""

def try_parse_json(raw_text):
    """Attempt to recover JSON even if model output is messy."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None

all_snippets = []
output_file = "hallucination_code_dataset.csv"

for i in range(100):
    chosen_lang = random.choice(languages)
    hallucination_type = random.choice(hallucination_types)
    model_name = models[i % len(models)]

    prompt = get_prompt_with_hallucination_type(chosen_lang, hallucination_type)
    print(f"\n🔹 Generating snippet {i+1} ({hallucination_type}) using {model_name}...")

    attempts = 0
    while attempts < 3:
        try:
            request_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a hallucination-focused code generator."},
                    {"role": "user", "content": prompt + "\nEnsure output is valid JSON with properly escaped code strings."}
                ],
                "temperature": 0.9,
                "max_tokens": 1024,
                "top_p": 0.9
            }

            response = client.chat.completions.create(**request_params)
            result = response.choices[0].message.content.strip()

            parsed = try_parse_json(result)
            if parsed:
                parsed["model"] = model_name
                normalized = {col: parsed.get(col, "") for col in COLUMNS}
                all_snippets.append(normalized)
                print(f"✅ Success: {hallucination_type}")
            else:
                print("⚠️ Invalid JSON. Raw output stored.")
                all_snippets.append({
                    "language": chosen_lang,
                    "hallucination_type": hallucination_type,
                    "raw_output": result,
                    "error": "Invalid JSON",
                    "model": model_name
                })

            time.sleep(10)
            break  # Success, move on
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            attempts += 1
            if "400" in str(e):
                print(f"⏭️ Skipping model {model_name} due to repeated 400 errors.")
                break
            print("Retrying in 30 seconds...")
            time.sleep(30)

# Save dataset
df_new = pd.DataFrame(all_snippets)
if os.path.exists(output_file):
    print(f"\n📂 '{output_file}' found — appending new data...")
    df_existing = pd.read_csv(output_file)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=["code_snippet"], inplace=True)
    df_combined.to_csv(output_file, index=False)
else:
    print(f"\n🆕 Creating new file '{output_file}'...")
    df_new.to_csv(output_file, index=False)

print(f"\n✅ Generation complete. Total snippets: {len(all_snippets)}")
