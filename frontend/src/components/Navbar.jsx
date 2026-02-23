import React from "react"
import { AppBar, Toolbar, Typography, Button, Box } from "@mui/material"
import { NavLink } from "react-router-dom"

export default function Navbar() {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Precolumbian Textil Generator
        </Typography>

        <Box sx={{ display: "flex", gap: 3 }}>
          <Button
            color="inherit"
            component={NavLink}
            to="/"
          >
            <Typography variant="h6">
              Home
            </Typography>
          </Button>

          <Button
            color="inherit"
            component={NavLink}
            to="/generacion"
          >
            <Typography variant="h6">
              Generación
            </Typography>
          </Button>

          <Button
            color="inherit"
            component={NavLink}
            to="/dataset"
          >
            <Typography variant="h6">
              Dataset
            </Typography>
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  )
}
