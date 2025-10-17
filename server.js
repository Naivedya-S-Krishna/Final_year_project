const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const { analyzeReport } = require("./suggestions");

const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "1mb" }));

app.get("/", (req, res) => res.send("Mental Health Chatbot API running ✅"));

app.post("/analyze-report", (req, res) => {
  try {
    const result = analyzeReport(req.body);
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
});

const PORT = 4000;
app.listen(PORT, () => console.log(`✅ Backend running at http://localhost:${PORT}`));

