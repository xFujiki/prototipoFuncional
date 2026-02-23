import { BrowserRouter, Routes, Route } from "react-router-dom"
import Navbar from "./components/Navbar"
import Home from "./pages/Home"
import Generacion from "./pages/Generacion"
import Dataset from "./pages/Dataset"

function App() {
  return (
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/generacion" element={<Generacion />} />
          <Route path="/dataset" element={<Dataset />} />
        </Routes>
      </BrowserRouter>
    )
}

export default App
