import os
import sys
import time
import uvicorn
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Get configuration from environment variables
HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", "8000"))

if __name__ == "__main__":
    # Wait for other services (like Weaviate) to be ready
    print("Waiting 5 seconds for services to be ready...")
    time.sleep(5)

    print(f"Starting backend server on {HOST}:{PORT}")
    uvicorn.run(
        "main:app", host=HOST, port=PORT, reload=False  # Disable reload in production
    )
