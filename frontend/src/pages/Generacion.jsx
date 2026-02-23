import React, { useState } from "react"
import EastIcon from "@mui/icons-material/East"
import {
  Container,
  Typography,
  Grid,
  Paper,
  Button,
  Box
} from "@mui/material"

export default function Generacion() {
  const [imageFile, setImageFile] = useState(null)
  const [preview, setPreview] = useState(null)

  const handleImageChange = (event) => {
    const file = event.target.files[0]
    if (file) {
      setImageFile(file)
      setPreview(URL.createObjectURL(file))
    }
  }

  const onClickGenerarImagen = async () => {
    if (!imageFile) return alert("Selecciona una imagen primero")

    const formData = new FormData()
    formData.append("image", imageFile)

    try {
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        body: formData
      })

      const data = await response.json()
      console.log("Respuesta backend:", data)

    } catch (error) {
      console.error("Error al generar imagen:", error)
    }
  }

  return (
    <Container sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Generación
      </Typography>

      <Grid container spacing={4}>
        {/* LADO IZQUIERDO: SUBIR IMAGEN */}
        <Grid item xs={12} md={5}>
          <Paper
            elevation={3}
            sx={{
              p: 3,
              height: "300px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
              //border: "2px dashed #ccc"
            }}
          >
            {!preview && (
              <Typography variant="h6" gutterBottom>
              Suba su imagen
            </Typography>
            )}

            {preview && (
              <Box
                component="img"
                src={preview}
                alt="Preview"
                sx={{
                  maxWidth: "100%",
                  maxHeight: "180px",
                  mb: 2,
                  borderRadius: 1
                }}
              />
            )}

            <Button
              variant="contained"
              component="label"
            >
              Seleccionar archivo
              <input
                type="file"
                accept="image/*"
                hidden
                onChange={handleImageChange}
              />
            </Button>
          </Paper>
        </Grid>

        {/* FLECHA */}
        <Grid item xs={12} md={1}>
          <Box
            sx={{
              height: "300px",
              display: "flex",
              justifyContent: "center",
              alignItems: "center"
            }}
          >
            <Button onClick={onClickGenerarImagen}>
              Generar
              <EastIcon fontSize="small" sx={{ml:1}}/>
            </Button>
          </Box>
        </Grid>

        {/* LADO DERECHO: RESULTADO */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={3}
            sx={{
              p: 3,
              height: "300px",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              color: "text.secondary"
            }}
          >
            <Typography>
              Imagen generada
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  )
}
