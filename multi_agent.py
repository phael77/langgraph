from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

try:
    if api_key is not None:
        print("API key is loaded sucessfully")
except Exception as e:
    print(f"Erro {e}")