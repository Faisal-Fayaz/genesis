/* Genesis gallery.
 *
 * The parameter form is generated from the generator schemas the server sends,
 * so adding a parameter to a generator makes it appear here with no change to
 * this file. Everything else is plain fetch and DOM work.
 */

const state = {
  generators: [],
  palettes: {},
  pieces: [],
  current: null,
  post: null,
};

const $ = (id) => document.getElementById(id);

function status(message, kind = "") {
  const el = $("status");
  el.textContent = message;
  el.className = `status ${kind}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

// -- form construction ---------------------------------------------------

function makeControl(spec) {
  const wrap = document.createElement("div");
  wrap.className = "param";

  const label = document.createElement("label");
  const name = document.createElement("span");
  name.textContent = spec.label || spec.name;
  const value = document.createElement("span");
  value.className = "value";
  label.append(name, value);
  wrap.append(label);

  let input;
  if (spec.kind === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(spec.default);
    value.textContent = input.checked ? "on" : "off";
    input.addEventListener("change", () => {
      value.textContent = input.checked ? "on" : "off";
    });
  } else if (spec.kind === "enum") {
    input = document.createElement("select");
    for (const choice of spec.choices || []) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice;
      option.selected = choice === spec.default;
      input.append(option);
    }
    input.addEventListener("change", () => {
      value.textContent = input.value;
    });
  } else if (spec.kind === "int" || spec.kind === "float") {
    input = document.createElement("input");
    input.type = "range";
    input.min = spec.minimum ?? 0;
    input.max = spec.maximum ?? (spec.kind === "int" ? 100 : 1);
    input.step = spec.step ?? (spec.kind === "int" ? 1 : 0.01);
    input.value = spec.default;
    const show = () => {
      value.textContent = input.value;
    };
    input.addEventListener("input", show);
    show();
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.value = spec.default ?? "";
    input.addEventListener("input", () => {
      value.textContent = input.value;
    });
  }

  input.dataset.name = spec.name;
  input.dataset.kind = spec.kind;
  wrap.append(input);

  if (spec.help) {
    const help = document.createElement("div");
    help.className = "help";
    help.textContent = spec.help;
    wrap.append(help);
  }
  return wrap;
}

function buildParams(schema) {
  const grid = $("param-grid");
  grid.replaceChildren();
  for (const spec of schema) {
    grid.append(makeControl(spec));
  }
}

function buildPost(schema) {
  const grid = $("post-grid");
  grid.replaceChildren();
  for (const spec of schema) {
    grid.append(makeControl(spec));
  }
}

function readControls(container) {
  const values = {};
  for (const input of container.querySelectorAll("[data-name]")) {
    const { name, kind } = input.dataset;
    if (kind === "bool") {
      values[name] = input.checked;
    } else if (kind === "int") {
      values[name] = parseInt(input.value, 10);
    } else if (kind === "float") {
      values[name] = parseFloat(input.value);
    } else {
      values[name] = input.value;
    }
  }
  return values;
}

function randomizeParams() {
  for (const input of $("param-grid").querySelectorAll("[data-name]")) {
    const { kind } = input.dataset;
    if (kind === "bool") {
      input.checked = Math.random() < 0.5;
      input.dispatchEvent(new Event("change"));
    } else if (kind === "enum") {
      const options = [...input.options];
      input.value = options[Math.floor(Math.random() * options.length)].value;
      input.dispatchEvent(new Event("change"));
    } else if (kind === "int" || kind === "float") {
      const min = parseFloat(input.min);
      const max = parseFloat(input.max);
      const value = min + Math.random() * (max - min);
      input.value = kind === "int" ? Math.round(value) : value.toFixed(4);
      input.dispatchEvent(new Event("input"));
    }
  }
}

// -- library ------------------------------------------------------------

async function loadLibrary() {
  const params = new URLSearchParams();
  const generator = $("filter-generator").value;
  const palette = $("filter-palette").value;
  if (generator) params.set("generator", generator);
  if (palette) params.set("palette", palette);
  if ($("filter-starred").checked) params.set("starred", "1");

  const payload = await api(`/api/pieces?${params}`);
  state.pieces = payload.pieces;
  $("library-empty").hidden = payload.pieces.length > 0;

  const container = $("pieces");
  container.replaceChildren();
  for (const piece of payload.pieces) {
    container.append(pieceCard(piece));
  }
}

function pieceCard(piece) {
  const card = document.createElement("div");
  card.className = "piece";
  card.dataset.digest = piece.digest;
  if (state.current?.digest === piece.digest) card.classList.add("selected");

  const img = document.createElement("img");
  img.src = piece.preview || piece.image;
  img.loading = "lazy";
  img.alt = `${piece.generator} ${piece.palette_name}`;
  card.append(img);

  const star = document.createElement("button");
  star.className = `star${piece.starred ? " on" : ""}`;
  star.textContent = piece.starred ? "★" : "☆";
  star.title = piece.starred ? "Unstar" : "Star";
  star.addEventListener("click", async (event) => {
    event.stopPropagation();
    const result = await api("/api/star", {
      method: "POST",
      body: JSON.stringify({ digest: piece.digest }),
    });
    piece.starred = result.starred;
    star.className = `star${result.starred ? " on" : ""}`;
    star.textContent = result.starred ? "★" : "☆";
  });
  card.append(star);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = `<span>${piece.generator}</span><span>${piece.palette_name}</span>`;
  card.append(meta);

  card.addEventListener("click", () => selectPiece(piece));
  return card;
}

async function selectPiece(piece) {
  state.current = piece;
  for (const card of document.querySelectorAll(".piece")) {
    card.classList.toggle("selected", card.dataset.digest === piece.digest);
  }

  const recipe = piece.recipe;
  $("generator").value = recipe.generator;
  $("width").value = recipe.width;
  $("height").value = recipe.height;
  $("seed").value = recipe.seed ?? "";
  $("palette").value = piece.palette_name;
  $("theme").checked = recipe.theme === "light";
  onGeneratorChanged();

  // Fill the controls with this recipe's resolved values.
  const schema = currentGenerator()?.params || [];
  for (const spec of schema) {
    const input = document.querySelector(`#param-grid [data-name="${spec.name}"]`);
    if (!input) continue;
    const value = recipe.params[spec.name];
    if (spec.kind === "bool") input.checked = Boolean(value);
    else input.value = value;
    input.dispatchEvent(new Event(input.dataset.kind === "int" || input.dataset.kind === "float" ? "input" : "change"));
  }
  for (const spec of state.post?.post || []) {
    const input = document.querySelector(`#post-grid [data-name="${spec.name}"]`);
    if (!input) continue;
    const value = (recipe.post || {})[spec.name] ?? spec.default;
    if (spec.kind === "bool") input.checked = Boolean(value);
    else input.value = value;
    input.dispatchEvent(new Event(input.dataset.kind === "int" || input.dataset.kind === "float" ? "input" : "change"));
  }

  showPreview(piece.image, piece);
}

// -- preview and render --------------------------------------------------

function showPreview(src, piece) {
  const wrap = $("preview");
  wrap.replaceChildren();

  const img = document.createElement("img");
  img.src = src;
  img.alt = "preview";
  wrap.append(img);

  if (piece?.digest) {
    const actions = document.createElement("div");
    actions.className = "apply-actions";

    const apply = document.createElement("button");
    apply.className = "primary";
    apply.textContent = "Set as wallpaper";
    apply.addEventListener("click", async () => {
      status("Setting wallpaper…", "busy");
      try {
        await api("/api/apply", {
          method: "POST",
          body: JSON.stringify({ digest: piece.digest }),
        });
        status("Wallpaper set");
      } catch (err) {
        status(err.message, "error");
      }
    });

    actions.append(apply);
    wrap.append(actions);
  }
}

function collectRecipe() {
  const recipe = {
    generator: $("generator").value,
    params: readControls($("param-grid")),
    width: parseInt($("width").value, 10),
    height: parseInt($("height").value, 10),
    palette_name: $("palette").value,
    theme: $("theme").checked ? "light" : "dark",
    seed: $("seed").value.trim() === "" ? null : $("seed").value.trim(),
    post: readControls($("post-grid")),
  };
  return recipe;
}

async function renderRecipe(preview = true) {
  const recipe = collectRecipe();
  if (preview) recipe.preview = true;

  status("Rendering…", "busy");
  const started = performance.now();
  try {
    const payload = await api("/api/render", {
      method: "POST",
      body: JSON.stringify(recipe),
    });
    state.current = payload.piece;
    const ms = Math.round(performance.now() - started);
    $("render-time").textContent = `${ms} ms`;
    showPreview(payload.piece.preview || payload.piece.image, payload.piece);
    status("Rendered");
    await loadLibrary();
  } catch (err) {
    status(err.message, "error");
  }
}

// -- wiring -------------------------------------------------------------

function currentGenerator() {
  return state.generators.find((g) => g.name === $("generator").value);
}

function onGeneratorChanged() {
  const generator = currentGenerator();
  if (generator) buildParams(generator.params);
}

async function init() {
  state.generators = await api("/api/generators");
  state.palettes = await api("/api/palettes");

  const generatorSelect = $("generator");
  const paletteSelect = $("palette");
  for (const generator of state.generators) {
    const option = document.createElement("option");
    option.value = generator.name;
    option.textContent = `${generator.name} (${generator.family})`;
    generatorSelect.append(option);
  }
  for (const name of Object.keys(state.palettes).sort()) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    paletteSelect.append(option);

    const filter = document.createElement("option");
    filter.value = name;
    filter.textContent = name;
    $("filter-palette").append(filter);
  }
  for (const generator of state.generators) {
    const filter = document.createElement("option");
    filter.value = generator.name;
    filter.textContent = generator.name;
    $("filter-generator").append(filter);
  }

  $("width").value = 1920;
  $("height").value = 1080;
  $("palette").value = "tide";

  const postSchema = await api("/api/post-schema").catch(() => null);
  state.post = postSchema;
  if (postSchema) buildPost(postSchema.post);

  generatorSelect.addEventListener("change", onGeneratorChanged);
  $("recipe-form").addEventListener("submit", (event) => {
    event.preventDefault();
    renderRecipe();
  });
  $("shuffle").addEventListener("click", randomizeParams);
  $("random").addEventListener("click", async () => {
    $("generator").value = state.generators[Math.floor(Math.random() * state.generators.length)].name;
    $("palette").value = Object.keys(state.palettes).sort()[Math.floor(Math.random() * Object.keys(state.palettes).length)];
    onGeneratorChanged();
    randomizeParams();
    await renderRecipe();
  });
  $("copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText(JSON.stringify(collectRecipe(), null, 2));
    status("Recipe copied");
  });
  for (const id of ["filter-generator", "filter-palette", "filter-starred"]) {
    $(id).addEventListener("change", loadLibrary);
  }

  onGeneratorChanged();
  await loadLibrary();
}

init().catch((err) => status(err.message, "error"));