import uvicorn
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.api_packer import ApiPacker

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "TEST_API"}


class Cuboid(BaseModel):
    width: float
    height: float
    depth: float


class RequestBody(BaseModel):
    cuboids: list[Cuboid]

@app.post("/packit")
async def pack_cubes(request_data: RequestBody):
    print(request_data)
    cuboids = request_data.cuboids
    if cuboids is None:
        raise HTTPException(status_code=400, detail="No cuboids provided")
    api_packer = ApiPacker(cuboids)
    result = api_packer.pack_cuboids()
    if result is None:
        raise HTTPException(status_code=500, detail="Can't pack the cuboids")
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
