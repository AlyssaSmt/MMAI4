from fastapi import FastAPI, Form
from open_vocab import recognize

app = FastAPI()

@app.post("/recognize")
async def recognize_draw(image_base64: str = Form(...)):
    return {"results": recognize(image_base64)}
