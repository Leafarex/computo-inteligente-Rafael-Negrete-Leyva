const path = require("path");
const fs = require("fs");
const express = require("express");
const ort = require("onnxruntime-node");
const { parse } = require("csv-parse/sync");

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(express.static(path.join(__dirname, "public")));

const MODEL_PATH = path.join(__dirname, "game_pipeline.onnx");
const SCHEMA_PATH = path.join(__dirname, "onnx_schema.json");
const CSV_PATH = path.join(__dirname, "..", "notebook", "game.csv");


let session = null;
let inputMetaByName = {};
let schema = null;

function loadSchema() {
  return JSON.parse(fs.readFileSync(SCHEMA_PATH, "utf-8"));
}

function readCsvRows() {
  const csvText = fs.readFileSync(CSV_PATH, "utf-8");
  const records = parse(csvText, { columns: true, skip_empty_lines: true });
  return records;
}

function uniqSorted(arr) {
  return Array.from(new Set(arr.filter((x) => x !== null && x !== undefined && String(x).trim() !== "")))
    .map((x) => String(x))
    .sort((a, b) => a.localeCompare(b));
}

function buildDropdownsFromCsv(rows) {
  const col = (name) => rows.map((r) => r[name]);

  return {
    season_type: uniqSorted(col("season_type")),
    team_abbreviation_home: uniqSorted(col("team_abbreviation_home")),
    team_name_home: uniqSorted(col("team_name_home")),
    team_abbreviation_away: uniqSorted(col("team_abbreviation_away")),
    team_name_away: uniqSorted(col("team_name_away")),
  };
}

function toFloat32(value, fieldName) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    throw new Error(`Campo numerico invalido: ${fieldName}`);
  }
  return n;
}

function makeTensor(name, value) {
  const meta = inputMetaByName[name];
  if (!meta) throw new Error(`Input desconocido: ${name}`);


  const dims = [1, 1];

  if (meta.type === "string") {
    if (typeof value !== "string" || value.trim() === "") {
      throw new Error(`Campo string invalido: ${name}`);
    }
    return new ort.Tensor("string", [value], dims);
  }

  
  const num = toFloat32(value, name);
  return new ort.Tensor("float32", [num], dims);
}

async function init() {
  schema = loadSchema();

  session = await ort.InferenceSession.create(MODEL_PATH);


  const metaList = session.handler.inputMetadata; 
  inputMetaByName = {};
  for (const m of metaList) inputMetaByName[m.name] = m;

  console.log("Modelo cargado. Outputs:", session.outputNames);
}

app.get("/api/meta", (req, res) => {
  try {
    const rows = readCsvRows();
    const dropdowns = buildDropdownsFromCsv(rows);

  
    const defaults = {
      season_type: dropdowns.season_type[0] ?? "Regular Season",
      team_abbreviation_home: dropdowns.team_abbreviation_home[0] ?? "UNK",
      team_name_home: dropdowns.team_name_home[0] ?? "UNK",
      team_abbreviation_away: dropdowns.team_abbreviation_away[0] ?? "UNK",
      team_name_away: dropdowns.team_name_away[0] ?? "UNK",
      season_id: 21946,
    };

    res.json({
      dropdowns,
      defaults,
 
    });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post("/api/predict", async (req, res) => {
  try {
    if (!session) throw new Error("Modelo no inicializado");

    const x = req.body || {};

    const feeds = {};
    for (const name of session.inputNames) {
      if (!(name in x)) {
        
        const meta = inputMetaByName[name];
        feeds[name] = meta.type === "string"
          ? new ort.Tensor("string", ["UNK"], [1, 1])
          : new ort.Tensor("float32", [0.0], [1, 1]);
      } else {
        feeds[name] = makeTensor(name, x[name]);
      }
    }

    const out = await session.run(feeds);

  
    const label = out.label.data[0];
    const probs = out.probabilities.data; 

    const pL = probs[0];
    const pW = probs[1];

    res.json({
      label,
      pW,
      pL,
    });
  } catch (e) {
    res.status(400).json({ error: String(e.message || e) });
  }
});

const PORT = 3000;
init().then(() => {
  app.listen(PORT, () => {
    console.log(`Servidor listo: http://localhost:${PORT}`);
  });
}).catch((e) => {
  console.error("Fallo init:", e);
  process.exit(1);
});
