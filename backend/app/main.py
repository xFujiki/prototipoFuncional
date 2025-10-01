from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="ML Model API")

# Configurar CORS para conectar con frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "ML API funcionando!"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Aquí irá tu modelo de ML
    return {
        "filename": image.filename,
        "message": "Predicción funcionando - modelo por implementar"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) #0.0.0.0 acepta conexiones desde cualquier IP
    #8000 es el puerto donde corre el servidor