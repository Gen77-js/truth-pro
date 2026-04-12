from dotenv import load_dotenv
# Load environment variables (API keys)
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import api

app = FastAPI(title="TruthLens Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to TruthLens Pro API"}
