import os
from groq import AsyncGroq

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=groq_api_key) if groq_api_key else None

# For backward compatibility
instructor_client = groq_client 