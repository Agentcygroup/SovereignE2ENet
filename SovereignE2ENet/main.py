from fastapi import FastAPI
import uvicorn
from core.agent import run_agent as sovereign_logic

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.post("/run")
def run():
    sovereign_logic()
    return {"status": "executed"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8624)
