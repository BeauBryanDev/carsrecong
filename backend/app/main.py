from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="CarsTracker API",
    description="Machine Learning backend for ALPR and Vehicle Tracking",
    version="1.0.0",
)

# Define allowed origins for Cross-Origin Resource Sharing (CORS)
# This ensures the React/Vite frontend can securely communicate with the API
# from different local development ports.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Add the CORS middleware to the application pipeline
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", tags=["System"])
def health_check():
    """
    Simple health check endpoint to verify the API is running.
    Useful for Docker health checks and initial debugging.
    
    Returns:
        dict: A status dictionary indicating the service is operational.
    """
    return {
        "status": "active", 
        "service": "carsrecong_api",
        "message": "System is ready for ML inference"
    }
    