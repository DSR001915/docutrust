// DocuTrust frontend app logic.
// No build step, no framework -- vanilla JS talking to FastAPI's REST + WS routes.

const API_BASE = "";
const WS_URL = `ws://${window.location.host}/ws/query`;

const el = {
  statusBadge: document.getElementById("statusBadge"),
  statusDot: document.querySelector(".status-dot"),
  statusText: document.getElementById("statusText"),
  docCount: document.getElementById("docCount"),
  documentList: document.getElementById("documentList"),
  fileInput: document.getElementById("fileInput"),
  uploadStatus: document.getElementById("uploadStatus"),
  ledgerScroll: document.getElementById("ledgerScroll"),
  ledgerEmpty: document.getElementById("ledgerEmpty"),
  ledgerSublabel: document.getElementById("ledgerSublabel"),
  briefScroll: document.getElementById("briefScroll"),
  briefSublabel: document.getElementById("briefSublabel"),
  exhibitsCited: document.getElementById("exhibitsCited"),
  exhibitsCitedList: document.getElementById("exhibitsCitedList"),
  queryInput: document.getElementById("queryInput"),
  queryButton: document.getElementById("queryButton"),
};

let ws = null;
let currentSourceMap = []; // index (1-based) -> {document_name, page_number} for the current answer

// ---- Node -> stamp presentation -------------------------------------------
const NODE_PRESENTATION = {
  retrieve: { icon: "⌕", label: "Retrieving candidates", tone: "neutral" },
  grade_documents: { icon: "✓", label: "Grading relevance", tone: (d) => gradeTone(d.grade) },
  generate: { icon: "✓", label: "Generating answer", tone: "verified" },
  rewrite_query: { icon: "⟲", label: "Rewriting query", tone: "correcting" },
  web_fallback: { icon: "⚑", label: "Escalating to web search", tone: "escalated" },
  corrective_generate: { icon: "⟲", label: "Generating corrected answer", tone: "correcting" },
};

function gradeTone(grade) {
  if (grade === "correct") return "verified";
  if (grade === "ambiguous") return "correcting";
  return "escalated";
}

// ---- Status badge ----------------------------------------------------------
function setStatus(state, text) {
  el.statusDot.className = `status-dot ${state}`;
  el.statusText.textContent = text;
}

// ---- WebSocket lifecycle ----------------------------------------------------
function connectWebSocket() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => setStatus("online", "ready");
  ws.onclose = () => {
    setStatus("error", "disconnected — retrying…");
    setTimeout(connectWebSocket, 2000);
  };
  ws.onerror = () => setStatus("error", "connection error");

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "trace") {
      addLedgerEntry(message.node, message.data);
    } else if (message.type === "final") {
      renderFinalAnswer(message.data);
      setStatus("online", "ready");
      el.queryButton.disabled = false;
    } else if (message.type === "error") {
      addLedgerEntry("error", { detail: message.data });
      setStatus("error", "error");
      el.queryButton.disabled = false;
    }
  };
}

// ---- Ledger rendering --------------------------------------------------------
function clearLedger() {
  el.ledgerScroll.innerHTML = "";
}

function addLedgerEntry(node, data) {
  el.ledgerEmpty?.remove();

  const presentation = NODE_PRESENTATION[node] || { icon: "•", label: node, tone: "neutral" };
  const tone = typeof presentation.tone === "function" ? presentation.tone(data) : presentation.tone;

  const entry = document.createElement("div");
  entry.className = "ledger-entry";

  const stamp = document.createElement("div");
  stamp.className = `ledger-stamp ${tone}`;
  stamp.textContent = presentation.icon;

  const body = document.createElement("div");
  body.className = "ledger-body";

  const title = document.createElement("div");
  title.className = "ledger-title";
  title.textContent = presentation.label;
  body.appendChild(title);

  const detail = document.createElement("div");
  detail.className = "ledger-detail";
  detail.appendChild(buildDetailLine(node, data));
  body.appendChild(detail);

  entry.appendChild(stamp);
  entry.appendChild(body);
  el.ledgerScroll.appendChild(entry);
  el.ledgerScroll.scrollTop = el.ledgerScroll.scrollHeight;
}

function buildDetailLine(node, data) {
  const frag = document.createDocumentFragment();

  if (node === "retrieve") {
    frag.append(`"${truncate(data.query, 60)}" — ${data.candidates_found} candidates → top ${data.reranked_to}`);
  } else if (node === "grade_documents") {
    const line = document.createElement("div");
    line.append(`grade: ${data.grade.toUpperCase()}  (score ${data.top_score.toFixed(3)})`);
    line.appendChild(buildScoreBar(data.top_score, data.threshold_correct, data.threshold_ambiguous));
    frag.appendChild(line);
  } else if (node === "rewrite_query") {
    frag.append(`"${truncate(data.original_query, 40)}" → "${truncate(data.rewritten_query, 40)}"  (attempt ${data.iteration})`);
  } else if (node === "web_fallback") {
    frag.append(`internal KB insufficient — pulled ${data.results_found} external result(s)`);
  } else if (node === "generate" || node === "corrective_generate") {
    frag.append(`${data.citations_count} citation(s) attached to answer`);
  } else if (node === "error") {
    frag.append(data.detail);
  } else {
    frag.append(JSON.stringify(data));
  }

  return frag;
}

function buildScoreBar(score, correctThreshold, ambiguousThreshold) {
  const wrap = document.createElement("span");
  wrap.className = "ledger-score-bar";

  const track = document.createElement("span");
  track.className = "score-track";
  const fill = document.createElement("span");
  fill.className = "score-fill";
  fill.style.width = `${Math.min(100, score * 100)}%`;
  fill.style.background =
    score >= correctThreshold ? "var(--verified)" : score >= ambiguousThreshold ? "var(--correcting)" : "var(--escalated)";
  track.appendChild(fill);
  wrap.appendChild(track);
  return wrap;
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n) + "…" : str;
}

// ---- Brief (answer) rendering -------------------------------------------------
function renderFinalAnswer(data) {
  el.briefSublabel.textContent = data.used_web_fallback ? "corrected · web-assisted" : "verified";
  el.ledgerSublabel.textContent = data.used_web_fallback ? "correction applied" : "single pass";

  el.briefScroll.innerHTML = "";

  if (data.used_web_fallback) {
    const banner = document.createElement("div");
    banner.className = "fallback-banner";
    banner.textContent = "⟲ Internal documents were insufficient — this answer was corrected with a web search fallback.";
    el.briefScroll.appendChild(banner);
  }

  const textEl = document.createElement("div");
  textEl.className = "brief-text";
  textEl.innerHTML = linkifyCitations(data.answer);
  el.briefScroll.appendChild(textEl);

  if (data.citations && data.citations.length > 0) {
    el.exhibitsCited.hidden = false;
    el.exhibitsCitedList.innerHTML = "";
    data.citations.forEach((chunkId, i) => {
      const chip = document.createElement("span");
      chip.className = "exhibit-chip";
      chip.textContent = `[${i + 1}] ${chunkId.slice(0, 8)}…`;
      el.exhibitsCitedList.appendChild(chip);
    });
  } else {
    el.exhibitsCited.hidden = true;
  }
}

function linkifyCitations(text) {
  const escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return escaped.replace(/\[(\d+)\]/g, '<span class="citation-marker">$1</span>');
}

// ---- Document upload ----------------------------------------------------------
async function refreshDocumentList() {
  try {
    const res = await fetch(`${API_BASE}/api/documents`);
    const { documents } = await res.json();
    el.docCount.textContent = `${documents.length} document${documents.length === 1 ? "" : "s"}`;
    el.documentList.innerHTML = "";
    documents.forEach((doc) => {
      const item = document.createElement("li");
      item.className = "exhibit-item";
      item.innerHTML = `
        <span class="exhibit-icon">PDF</span>
        <span class="exhibit-meta">
          <span class="exhibit-name">${escapeHtml(doc.filename)}</span>
          <span class="exhibit-sub">${doc.page_count}p · ${doc.chunk_count} chunks</span>
        </span>`;
      el.documentList.appendChild(item);
    });
  } catch (err) {
    console.error("Failed to load documents", err);
  }
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

el.fileInput.addEventListener("change", async () => {
  const file = el.fileInput.files[0];
  if (!file) return;

  el.uploadStatus.className = "upload-status";
  el.uploadStatus.textContent = `Ingesting ${file.name}…`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }
    const result = await res.json();
    el.uploadStatus.className = "upload-status success";
    el.uploadStatus.textContent = `✓ ${result.chunk_count} chunks indexed`;
    await refreshDocumentList();
  } catch (err) {
    el.uploadStatus.className = "upload-status error";
    el.uploadStatus.textContent = `✗ ${err.message}`;
  } finally {
    el.fileInput.value = "";
  }
});

// ---- Query submission -----------------------------------------------------------
function submitQuery() {
  const query = el.queryInput.value.trim();
  if (!query) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    setStatus("error", "not connected");
    return;
  }

  clearLedger();
  el.ledgerSublabel.textContent = "running…";
  el.briefSublabel.textContent = "—";
  el.briefScroll.innerHTML = '<div class="brief-empty">Working…</div>';
  el.exhibitsCited.hidden = true;
  el.queryButton.disabled = true;
  setStatus("busy", "reasoning…");

  ws.send(JSON.stringify({ query }));
}

el.queryButton.addEventListener("click", submitQuery);
el.queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitQuery();
});

// ---- Init -------------------------------------------------------------------------
connectWebSocket();
refreshDocumentList();
