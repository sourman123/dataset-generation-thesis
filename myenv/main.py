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

languages = ["JavaScript"]
models = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b"
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
    # Integration/Deployment Hallucinations
    if hallucination_type == "Integration_Deployment_Hallucination":
        file_type = random.choice(["package.json", "Dockerfile", ".github/workflows/deploy.yml", ".env", "vercel.json"])
        return f"""
You are an expert full-stack JavaScript developer.

Generate a realistic **{file_type}** file (20–80 lines) for a Node.js + React project
that contains an **Integration/Deployment Hallucination** — invalid configurations, imaginary environment variables, or incorrect Docker/GitHub CI setups.

Examples:
- Fake config keys in package.json or vercel.json
- Invalid YAML keys or GitHub Action triggers
- Nonexistent Docker base images
- Unreal npm or build flags

Respond strictly in **valid JSON** with this structure:
{{
  "language": "{file_type}",
  "hallucination_type": "{hallucination_type}",
  "code_snippet": "<escaped contents of {file_type}>",
  "description": "<summary of what this configuration aims to do>",
  "hallucination_details": "<explain which parts are hallucinated>"
}}

Return only a single valid JSON object. Do not include markdown, comments, or text outside the JSON.
"""

    # Correct Code for configs + code
    elif hallucination_type == "Correct_Code":
        file_type = random.choice([
            "JavaScript",
            ".env",
            "package.json",
            "vercel.json",
            "Dockerfile",
            ".github/workflows/deploy.yml"
        ])
        return f"""
You are an expert full-stack JavaScript developer and DevOps engineer.

Generate a **realistic and correct {file_type}** for a Node.js + React project (20–80 lines).  
This file should represent secure, production-quality logic or configuration.  

Examples of valid cases:
- Correct `Dockerfile` for a Node.js app using `node:18-alpine`
- Valid `.env` with properly formatted secrets
- Realistic and secure `vercel.json` configuration
- Proper `.github/workflows/deploy.yml` workflow for CI/CD
- A secure Express route or React component

Avoid fake keys, invalid Docker images, or imaginary YAML/JSON fields.

Respond strictly in **valid JSON** with this structure:
{{
  "language": "{file_type}",
  "hallucination_type": "{hallucination_type}",
  "code_snippet": "<escaped contents of {file_type}>",
  "description": "<brief summary of what this file or code does>",
  "hallucination_details": "none"
}}

Return only a single valid JSON object. Do not include markdown, comments, or text outside the JSON.
"""

    # API misuse or security hallucinations
    else:
        return f"""
You are an expert full-stack {language} developer.

Generate a realistic {language} code snippet (20–80 lines) that represents the category: **{hallucination_type}**.

Definitions:
1. **API/Library Misuse** — incorrect or invented use of an API, library, or function.
   Example: calling non-existent Express.js functions, using a fake React hook, or misusing Mongoose methods.
2. **Security-Critical Hallucination** — insecure or fabricated logic related to authentication, encryption, or security.
   Example: comparing passwords as plain strings, using made-up crypto APIs, or unsafe JWT handling.
3. **Correct Code** — realistic, production-quality Node.js + React snippet that is well-structured and secure.

Requirements:
- Alternate between frontend (React) and backend (Node.js + Express) logic across generations.
- The code must look realistic (imports, functions, etc.), not toy-like.
- If it’s a hallucination, make it look plausible but wrong.
- If it’s correct, make it valid and secure.
- Respond strictly in **valid JSON**, no extra text or markdown.

Respond with this structure:
{{
  "language": "{language}",
  "hallucination_type": "{hallucination_type}",
  "code_snippet": "<escaped {language} code>",
  "description": "<brief summary>",
  "hallucination_details": "<if hallucinated: explain; if correct: 'none'>"
}}

Return only a single valid JSON object. Do not include markdown, comments, or text outside the JSON.
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

for i in range(200):  # adjust number of samples
    chosen_lang = random.choice(languages)
    hallucination_type = random.choices(
    hallucination_types,
    weights=[1, 1, 1, 3],
    k=1
    )[0]
    model_name = models[i % len(models)]

    print(f"\n🔹 Generating snippet {i+1} ({hallucination_type}) using {model_name}...")

    prompt = get_prompt_with_hallucination_type(chosen_lang, hallucination_type)
    attempts = 0

    while attempts < 3:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a hallucination-focused code generation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=1500,
                top_p=0.9,
                response_format={"type": "json_object"}
            )

            result = response.choices[0].message.content.strip()
            parsed = try_parse_json(result)

            if parsed:
                parsed["model"] = model_name
                normalized = {col: parsed.get(col, "") for col in COLUMNS}
                all_snippets.append(normalized)
                print(f"Success: {hallucination_type}")
            else:
                print("⚠️ Invalid JSON. Raw output saved.")
                all_snippets.append({
                    "language": chosen_lang,
                    "hallucination_type": hallucination_type,
                    "code_snippet": "",
                    "description": "",
                    "hallucination_details": "Invalid JSON",
                    "model": model_name
                })
                with open(f"raw_output_{i+1}.txt", "w", encoding="utf-8") as raw:
                    raw.write(result)

            time.sleep(8)
            break
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            attempts += 1
            if "400" in str(e):
                print(f"Skipping model {model_name} after repeated 400 errors.")
                break
            print("Retrying in 20 seconds...")
            time.sleep(20)

df_new = pd.DataFrame(all_snippets)
if os.path.exists(output_file):
    print(f"\n📂 '{output_file}' found — appending new data...")
    df_existing = pd.read_csv(output_file)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=["code_snippet"], inplace=True)
    df_combined.to_csv(output_file, index=False)
else:
    print(f"\nCreating new file '{output_file}'...")
    df_new.to_csv(output_file, index=False)

print(f"\nGeneration complete. Total snippets: {len(all_snippets)}")
