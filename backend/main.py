from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import asyncio

from digesting import digest_directory
from graph import send_chunks_to_graph

app = FastAPI()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

fileCounter = 0


@app.get("/")
def read_root():
    return {"message": "Welcome to the Base API!"}


@app.post("/upload-course/")
async def upload_course(file: UploadFile = File(...)):
    global fileCounter

    if not file.filename.endswith(".zip"):
        return {"error": "Only .zip files are allowed"}

    # Save uploaded file temporarily
    upload_path = UPLOAD_DIR / f"{fileCounter}.zip"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file.file.close()

    # Digest: extract and chunk
    target_dir = UPLOAD_DIR / str(fileCounter)
    target_dir.mkdir(parents=True, exist_ok=True)
    import zipfile
    with zipfile.ZipFile(upload_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    upload_path.unlink()  # remove zip

    chunks = digest_directory(target_dir)

    print(f"\n\n\nExtracted {len(chunks)} chunks from uploaded course.\n\n\n\n")

    # Send chunks to Graphiti
    await send_chunks_to_graph(chunks, source_description=f"Upload ID {fileCounter}")

    fileCounter += 1

    return {"upload_id": fileCounter - 1, "chunks_added": len(chunks)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
