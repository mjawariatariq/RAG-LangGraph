import google.generativeai as genai

# API key set karo
genai.configure(api_key="YOUR_API_KEY")

# Saare models list karo
for model in genai.list_models():
    if "embedContent" in model.supported_generation_methods:
        print(f"✅ {model.name}")