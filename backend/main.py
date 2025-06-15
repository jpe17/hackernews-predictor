"""
FastAPI application to serve the HackerNews Score Prediction model.
"""
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# Import the Scorer class from our prediction module
from .predict import Scorer

app = FastAPI(
    title="HackerNews Score Predictor API",
    description="An API to predict the potential score of a HackerNews submission.",
    version="1.0.0"
)

# Initialize the scorer once when the app starts to load the model into memory.
# This is more efficient than loading it on every request.
try:
    scorer = Scorer()
except FileNotFoundError as e:
    # If model artifacts are not found, we can't make predictions.
    # We'll initialize scorer to None and handle this gracefully in the endpoint.
    print(f"CRITICAL: Could not load model artifacts. The /predict endpoint will be disabled. Error: {e}")
    scorer = None

# Define the request body model using Pydantic for data validation
class Story(BaseModel):
    title: str
    url: str  # Using str as HttpUrl can be too strict for some HN links
    user: str
    timestamp: int = int(datetime.now().timestamp())

    class Config:
        schema_extra = {
            "example": {
                "title": "Show HN: I built a tool to predict HN scores",
                "url": "https://github.com/myuser/myproject",
                "user": "myuser",
                "timestamp": 1678886400
            }
        }

# Define the response body model
class PredictionResponse(BaseModel):
    predicted_score: int

@app.get("/", tags=["Root"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the HackerNews Score Predictor API!"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def post_predict(story: Story):
    """
    Predicts the score of a HackerNews story based on its features.
    
    - **title**: The title of the story.
    - **url**: The URL of the story. Can be empty for self-posts.
    - **user**: The username of the submitter.
    - **timestamp**: The UNIX timestamp of the submission.
    """
    if scorer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure training artifacts are available and the server is restarted."
        )
    
    # Convert UNIX timestamp to a datetime object, as required by the scorer
    submission_time = datetime.fromtimestamp(story.timestamp)
    
    # Use the loaded scorer to predict the score
    score = scorer.predict(
        title=story.title,
        url=story.url,
        user=story.user,
        submission_time=submission_time
    )
    return {"predicted_score": score}

# To run this app:
# uvicorn backend.main:app --reload --port 8888 