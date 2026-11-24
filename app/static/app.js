// HyM · frontend (sin frameworks). Habla con el servidor FastAPI (app/server.py).
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const GRUPOS_ES = {
  "Garment Upper body": "Parte de arriba", "Garment Lower body": "Parte de abajo", "Garment Full body": "Vestidos y conjuntos",
  "Accessories": "Accesorios", "Shoes": "Zapatos", "Swimwear": "Trajes de baño", "Underwear": "Ropa interior",
  "Socks & Tights": "Calcetines y medias", "Nightwear": "Pijamas", "Bags": "Bolsas", "Cosmetic": "Cosméticos",
  "Underwear/nightwear": "Ropa interior y pijamas",
};
const LINEAS_ES = { "": "Todo", mujer: "Mujer", hombre: "Hombre", divided: "Divided", ninos: "Niños", sport: "Sport" };
const SUGERENCIAS = ["Busco un vestido para una boda en la playa", "Una chamarra de cuero negra para hombre",
                     "¿Qué zapatos van con unos jeans negros?"];

const state = {
  sid: (() => { try { return localStorage.getItem("hym-sid") || ""; } catch { return ""; } })(),
  linea: "", grupo: "", pagina: 0, total: 0, archivo: null, enviando: false,
};

// ------------------------------------------------------------ estado del servidor / LLM
async function cargarEstado() {
  try {
    const r = await fetch("/api/estado").then(x => x.json());
    const el = $("#estado");
    if (!r.llm_revisado) {
      el.className = "estado"; el.textContent = "Conectando con la IA…";
      setTimeout(cargarEstado, 3000); return;
    }
    el.className = "estado " + (r.llm_ok ? "ok" : "off");
    el.textContent = r.llm_ok ? `IA conectada · ${r.llm_modelo}` : "IA sin conexión · modo básico";
    el.title = r.llm_ok ? r.llm_url : (r.llm_error || "");
    $("#chat-sub").textContent = r.llm_ok ? `HyM · ${r.llm_modelo}` : "HyM · modo básico (sin LLM)";
    $("#footer-llm").textContent = `LLM: ${r.llm_modelo}`;
    $("#stat-articulos").textContent = `${Math.round(r.articulos / 1000)}k`;
  } catch { $("#estado").textContent = "Servidor sin conexión"; }
}

// ------------------------------------------------------------ catálogo
function tarjeta(a) {
  const el = $("#tpl-card").content.firstElementChild.cloneNode(true);
  const img = $("img", el);
  img.src = a.img; img.alt = a.prod_name;
  $(".card-name", el).textContent = a.prod_name;
  $(".card-meta", el).textContent = `${a.product_type_name} · ${a.colour_group_name}`;
  $('[data-accion="similar"]', el).onclick = () => preguntar("Muéstrame prendas parecidas a esta", a);
  $('[data-accion="combinar"]', el).onclick = () => preguntar("¿Con qué puedo combinar esta prenda?", a);
  return el;
}

async function cargarCatalogo(reiniciar = true) {
  const grid = $("#grid");
  if (reiniciar) {
    state.pagina = 0;
    grid.innerHTML = "";
    for (let i = 0; i < 8; i++) grid.insertAdjacentHTML("beforeend",
      '<article class="card skeleton"><div class="card-img"></div><h3 class="card-name">&nbsp;</h3></article>');
  }
  const q = new URLSearchParams({ linea: state.linea, grupo: state.grupo, pagina: state.pagina, n: 24 });
  const r = await fetch("/api/catalogo?" + q).then(x => x.json());
  $$(".skeleton", grid).forEach(x => x.remove());
  r.articulos.forEach(a => grid.appendChild(tarjeta(a)));
  state.total = r.total;
  $("#more").hidden = grid.children.length >= r.total;
  const linea = LINEAS_ES[state.linea], grupo = GRUPOS_ES[state.grupo] || state.grupo;
  $("#crumbs").textContent = ["HyM", linea, grupo].filter(Boolean).join(" / ");
  $("#catalog-title").textContent = grupo ? `${grupo}${state.linea ? " · " + linea : ""}`
                                          : (state.linea ? `${linea}: lo más vendido` : "Lo más vendido");
}

async function cargarGrupos() {
  const r = await fetch("/api/grupos?linea=" + state.linea).then(x => x.json());
  const ul = $("#grupos");
  ul.innerHTML = "";
  const item = (g, txt, n) => {
    const li = document.createElement("li"), a = document.createElement("a");
    a.href = "#catalogo"; a.textContent = txt;
    if (n != null) { const s = document.createElement("small"); s.textContent = n.toLocaleString("es-MX"); a.append(" ", s); }
    a.classList.toggle("on", state.grupo === g);
    a.onclick = e => { e.preventDefault(); state.grupo = g; cargarGrupos(); cargarCatalogo(); };
    li.appendChild(a); ul.appendChild(li);
  };
  item("", "Todas las categorías");
  r.forEach(x => item(x.grupo, GRUPOS_ES[x.grupo] || x.grupo, x.n));
}

function elegirLinea(linea) {
  state.linea = linea; state.grupo = "";
  $$("[data-linea]").forEach(a => a.classList.toggle("on", a.dataset.linea === linea && linea !== ""));
  $$("#lineas a").forEach(a => a.classList.toggle("on", a.dataset.linea === linea));
  cargarGrupos(); cargarCatalogo();
}

// ------------------------------------------------------------ chat
function abrirChat(pedirFoto = false) {
  $("#chat").classList.add("open"); $("#chat").setAttribute("aria-hidden", "false");
  $("#overlay").hidden = false;
  if (!$("#mensajes").children.length) bienvenida();
  if (pedirFoto) $("#archivo").click(); else setTimeout(() => $("#texto").focus(), 300);
}
function cerrarChat() {
  $("#chat").classList.remove("open"); $("#chat").setAttribute("aria-hidden", "true");
  $("#overlay").hidden = true;
}

function burbujaBot(texto, clase = "") {
  const div = document.createElement("div");
  div.className = "msg bot " + clase;
  const p = document.createElement("p"); p.className = "texto"; p.textContent = texto;
  div.appendChild(p);
  $("#mensajes").appendChild(div);
  return div;
}

function bienvenida() {
  const d = burbujaBot("¡Hola! Soy tu estilista. Sube la foto de una prenda y te digo qué hay parecido y con qué " +
                       "combinarla, o cuéntame qué buscas.");
  const chips = document.createElement("div"); chips.className = "chips";
  SUGERENCIAS.forEach(s => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = s;
    b.onclick = () => enviar(s);
    chips.appendChild(b);
  });
  const foto = document.createElement("button"); foto.type = "button"; foto.textContent = "📷 Subir una foto";
  foto.onclick = () => $("#archivo").click();
  chips.appendChild(foto);
  d.appendChild(chips);
}

function burbujaUsuario(texto, imgSrc) {
  const div = document.createElement("div"); div.className = "msg user";
  if (imgSrc) { const im = document.createElement("img"); im.src = imgSrc; im.alt = "Foto enviada"; div.appendChild(im); }
  if (texto) { const p = document.createElement("div"); p.textContent = texto; div.appendChild(p); }
  $("#mensajes").appendChild(div);
}

function tarjetaRec(a) {
  const el = document.createElement("article"); el.className = "rec";
  const img = document.createElement("img"); img.src = a.img; img.alt = a.prod_name; img.loading = "lazy";
  img.title = "¿Con qué combino esta prenda?";
  img.onclick = () => preguntar("¿Con qué puedo combinar esta prenda?", a);
  const body = document.createElement("div"); body.className = "rec-body";
  const tag = document.createElement("span");
  tag.className = "tag " + a.tipo; tag.textContent = a.tipo === "similar" ? "Alternativa" : "Para combinar";
  const h = document.createElement("h4"); h.textContent = a.prod_name;
  const meta = document.createElement("p"); meta.className = "meta"; meta.textContent = `${a.product_type_name} · ${a.colour_group_name}`;
  const mot = document.createElement("p"); mot.className = "motivo"; mot.textContent = a.motivo;
  body.append(tag, h, meta, mot);
  const mini = document.createElement("div"); mini.className = "mini";
  [["Parecidas", "Muéstrame prendas parecidas a esta"], ["Combinar", "¿Con qué puedo combinar esta prenda?"]].forEach(([t, q]) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = t;
    b.onclick = () => preguntar(q, a);
    mini.appendChild(b);
  });
  el.append(img, body, mini);
  return el;
}

function detallesTecnicos(r) {
  const d = document.createElement("details"); d.className = "tech";
  const s = document.createElement("summary"); s.textContent = "Detalles técnicos";
  const pre = document.createElement("pre");
  const p = r.plan || {};
  const lat = Object.entries(r.latencias || {}).map(([k, v]) => `${k} ${v}s`).join(" · ");
  pre.textContent = [
    p.intencion ? `intención: ${p.intencion}` : null,
    p.consulta_en ? `consulta (Fashion-CLIP): «${p.consulta_en}»` : null,
    p.color ? `color: ${p.color}` : null,
    (p.excluir_color || []).length ? `excluir color: ${p.excluir_color.join(", ")}` : null,
    p.linea ? `línea: ${p.linea}` : null,
    p.tipo_complemento ? `tipo de complemento: ${p.tipo_complemento}` : null,
    r.flags?.length ? `avisos: ${r.flags.join(", ")}` : null,
    `latencia: ${lat}`,
  ].filter(Boolean).join("\n");
  d.append(s, pre);
  return d;
}

function escribiendo() {
  const div = document.createElement("div"); div.className = "msg bot"; div.id = "typing";
  div.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>' +
                  '<p class="typing-note">El estilista está buscando en el catálogo…</p>';
  $("#mensajes").appendChild(div);
  scrollChat();
}
function scrollChat() { const b = $("#mensajes"); b.scrollTop = b.scrollHeight; }

async function enviar(texto, articulo = null) {
  if (state.enviando) return;
  texto = (texto ?? $("#texto").value).trim();
  const archivo = articulo ? null : state.archivo;
  if (!texto && !archivo && !articulo) return;
  abrirChat();
  state.enviando = true; $("#enviar").disabled = true;
  burbujaUsuario(texto, articulo ? articulo.img : (archivo ? URL.createObjectURL(archivo) : null));
  $("#texto").value = ""; autoAltura(); quitarAdjunto();
  escribiendo();

  const fd = new FormData();
  fd.append("session_id", state.sid);
  fd.append("mensaje", texto);
  if (articulo) fd.append("article_id", articulo.article_id);
  if (archivo) fd.append("imagen", archivo);
  try {
    const res = await fetch("/api/chat", { method: "POST", body: fd });
    const r = await res.json();
    $("#typing")?.remove();
    if (!res.ok) throw new Error(r.detail || "Error del servidor");
    state.sid = r.session_id;
    try { localStorage.setItem("hym-sid", state.sid); } catch {}
    const d = burbujaBot(r.mensaje);
    if (r.articulos.length) {
      const recs = document.createElement("div"); recs.className = "recs";
      r.articulos.forEach(a => recs.appendChild(tarjetaRec(a)));
      d.appendChild(recs);
    }
    d.appendChild(detallesTecnicos(r));
    if (!r.llm_ok) cargarEstado();
  } catch (e) {
    $("#typing")?.remove();
    burbujaBot("No pude completar la búsqueda: " + e.message, "error");
  } finally {
    state.enviando = false; $("#enviar").disabled = false;
    // muestra el inicio de la respuesta (el texto del estilista), no el final de las tarjetas
    const ultimo = $$("#mensajes .msg.bot").pop();
    if (ultimo) ultimo.scrollIntoView({ behavior: "smooth", block: "start" }); else scrollChat();
  }
}

function preguntar(texto, articulo) { abrirChat(); enviar(texto, articulo); }

// ------------------------------------------------------------ foto adjunta
function adjuntar(file) {
  if (!file || !file.type.startsWith("image/")) return;
  state.archivo = file;
  $("#adjunto-img").src = URL.createObjectURL(file);
  $("#adjunto").hidden = false;
  abrirChat();
  $("#texto").placeholder = "¿Qué quieres saber de esta prenda? (opcional)";
  $("#texto").focus();
}
function quitarAdjunto() {
  state.archivo = null; $("#adjunto").hidden = true; $("#archivo").value = "";
  $("#texto").placeholder = "Escribe o sube una foto…";
}
function autoAltura() { const t = $("#texto"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 120) + "px"; }

// ------------------------------------------------------------ eventos
$$("[data-open-chat]").forEach(b => b.addEventListener("click", () => abrirChat(b.dataset.openChat === "foto")));
$("#cerrar").onclick = cerrarChat;
$("#overlay").onclick = cerrarChat;
document.addEventListener("keydown", e => { if (e.key === "Escape") cerrarChat(); });
$("#nueva").onclick = async () => {
  const fd = new FormData(); fd.append("session_id", state.sid);
  fetch("/api/reiniciar", { method: "POST", body: fd });
  state.sid = ""; try { localStorage.removeItem("hym-sid"); } catch {}
  $("#mensajes").innerHTML = ""; quitarAdjunto(); bienvenida();
};
$("#chat-form").onsubmit = e => { e.preventDefault(); enviar(); };
$("#texto").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); enviar(); } });
$("#texto").addEventListener("input", autoAltura);
$("#archivo").onchange = e => adjuntar(e.target.files[0]);
$("#quitar").onclick = quitarAdjunto;
$("#buscar").onsubmit = e => { e.preventDefault(); const q = e.target.q.value; e.target.q.value = ""; if (q.trim()) enviar(q); };
document.addEventListener("paste", e => {
  const f = [...(e.clipboardData?.files || [])].find(x => x.type.startsWith("image/"));
  if (f) { e.preventDefault(); adjuntar(f); }
});
const chat = $("#chat");
["dragenter", "dragover"].forEach(t => chat.addEventListener(t, e => { e.preventDefault(); chat.classList.add("drag"); }));
["dragleave", "drop"].forEach(t => chat.addEventListener(t, e => { e.preventDefault(); chat.classList.remove("drag"); }));
chat.addEventListener("drop", e => adjuntar(e.dataTransfer.files[0]));
$$("[data-linea]").forEach(a => a.addEventListener("click", () => elegirLinea(a.dataset.linea)));
$("#more").onclick = () => { state.pagina++; cargarCatalogo(false); };

cargarEstado(); setInterval(cargarEstado, 30000);
cargarGrupos(); cargarCatalogo();
