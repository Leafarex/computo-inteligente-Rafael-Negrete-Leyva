async function getMeta() {
  const r = await fetch("/api/meta");
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Error meta");
  return j;
}

function fillSelect(el, values, selected) {
  el.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === selected) opt.selected = true;
    el.appendChild(opt);
  }
}

function createNumberInput(name) {
  const label = document.createElement("label");
  label.innerHTML = `${name}<input id="num_${name}" type="number" step="any" placeholder="0" />`;
  return label;
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

async function predict(payload) {
  const r = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Error predict");
  return j;
}

(async function main() {
  const season_type = document.getElementById("season_type");
  const team_abbreviation_home = document.getElementById("team_abbreviation_home");
  const team_name_home = document.getElementById("team_name_home");
  const team_abbreviation_away = document.getElementById("team_abbreviation_away");
  const team_name_away = document.getElementById("team_name_away");
  const season_id = document.getElementById("season_id");

  const numericWrap = document.getElementById("numericFields");
  const btn = document.getElementById("btnPredict");
  const result = document.getElementById("result");
  const error = document.getElementById("error");

  hide(result); hide(error);
  btn.disabled = true;

  let meta;
  try {
    meta = await getMeta();
  } catch (e) {
    error.textContent = String(e.message || e);
    show(error);
    return;
  }

  fillSelect(season_type, meta.dropdowns.season_type, meta.defaults.season_type);
  fillSelect(team_abbreviation_home, meta.dropdowns.team_abbreviation_home, meta.defaults.team_abbreviation_home);
  fillSelect(team_name_home, meta.dropdowns.team_name_home, meta.defaults.team_name_home);
  fillSelect(team_abbreviation_away, meta.dropdowns.team_abbreviation_away, meta.defaults.team_abbreviation_away);
  fillSelect(team_name_away, meta.dropdowns.team_name_away, meta.defaults.team_name_away);

  season_id.value = meta.defaults.season_id;


  const numericFields = meta.numericFields.filter((n) => n !== "season_id");
  for (const name of numericFields) {
    numericWrap.appendChild(createNumberInput(name));
  }

  btn.disabled = false;

  btn.addEventListener("click", async () => {
    hide(result); hide(error);
    btn.disabled = true;

    try {
      const payload = {
        season_type: season_type.value,
        team_abbreviation_home: team_abbreviation_home.value,
        team_name_home: team_name_home.value,
        team_abbreviation_away: team_abbreviation_away.value,
        team_name_away: team_name_away.value,
        season_id: Number(season_id.value),
      };

     
      for (const name of numericFields) {
        const el = document.getElementById(`num_${name}`);
        const v = el.value.trim();
        if (v !== "") payload[name] = Number(v);
      }

   
      if (!Number.isFinite(payload.season_id)) {
        throw new Error("season_id invalido");
      }

      const out = await predict(payload);

      result.innerHTML = `
        <h2>Resultado</h2>
        <div>Prediccion: <b>${out.label}</b></div>
        <div>P(W): ${Number(out.pW).toFixed(6)}</div>
        <div>P(L): ${Number(out.pL).toFixed(6)}</div>
      `;
      show(result);
    } catch (e) {
      error.textContent = String(e.message || e);
      show(error);
    } finally {
      btn.disabled = false;
    }
  });
})();
