from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

from llm import chat, memory_clear

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str




@app.post("/chat")
async def send_message(data : Message, response_class=None, response_content_type="text/plain"):
    res = chat(data.message,)
    return str(res)


@app.post("/clear")
async def send_messages(response_class=None, response_content_type="text/plain"):
    memory_clear()
    return str("cleared")


