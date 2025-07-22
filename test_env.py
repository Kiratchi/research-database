from dotenv import load_dotenv
import os

load_dotenv()

print("ES_HOST:", os.getenv("ES_HOST"))
print("ES_USER:", os.getenv("ES_USER")) 
print("LITELLM_API_KEY:", os.getenv("LITELLM_API_KEY"))
