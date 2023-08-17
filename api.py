import uvicorn
import json

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "TEST_API"}


class Point(BaseModel):
    x: float
    y: float
    z: float


class RequestBody(BaseModel):
    points: list[Point]

@app.post("/packit")
async def pack_cubes(request_data: RequestBody):
    points = request_data.points
    print(points)
    with open("data/test_api_data.json", "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
