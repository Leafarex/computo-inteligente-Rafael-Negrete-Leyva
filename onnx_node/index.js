const ort = require("onnxruntime-node");

function s(name) {
  if (name === "season_type") return "Regular Season";
  if (name === "team_abbreviation_home") return "HUS";
  if (name === "team_name_home") return "Toronto Huskies";
  if (name === "team_abbreviation_away") return "NYK";
  if (name === "team_name_away") return "New York Knicks";
  return "UNK";
}

function f(name) {
  if (name === "season_id") return 21946.0;
  return 0.0;
}

async function main() {
  const session = await ort.InferenceSession.create("./game_pipeline.onnx");

  const metaList = session.handler.inputMetadata;
  const inputNames = session.inputNames;

  const metaByName = {};
  for (const m of metaList) metaByName[m.name] = m;

  const feeds = {};
  for (const name of inputNames) {
    const t = metaByName[name].type;
    const dims = [1, 1];
    if (t === "string") feeds[name] = new ort.Tensor("string", [s(name)], dims);
    else feeds[name] = new ort.Tensor("float32", [f(name)], dims);
  }

  const results = await session.run(feeds);

  console.log("Outputs:", Object.keys(results));
  console.log("Prediccion:", results.label.data[0]);

  const probs = results.probabilities.data;
  const dims = results.probabilities.dims;
  console.log("Probabilidades dims:", dims);
  console.log("Probabilidades data:", probs);
}

main().catch((e) => {
  console.error("ERROR:", e);
  process.exit(1);
});
