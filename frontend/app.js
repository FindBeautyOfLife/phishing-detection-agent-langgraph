const state = { lastResults: null, busy: false, assets: new Map() };
const messages = document.querySelector("#messages");
const input = document.querySelector("#promptInput");
const sendButton = document.querySelector("#sendButton");
const resultsContent = document.querySelector("#resultsContent");
const resultCount = document.querySelector("#resultCount");
const assetsContent = document.querySelector("#assetsContent");
const toast = document.querySelector("#toast");
const collapseMessagesButton = document.querySelector("#collapseMessages");
const toggleAssetsButton = document.querySelector("#toggleAssets");

function addMessage(text, role = "assistant", loading = false) {
  const row = document.createElement("div");
  row.className = `message ${role}`;
  row.innerHTML = `<div class="avatar">${role === "user" ? "我" : "盾"}</div><div class="bubble">${loading ? '<div class="loading"><i class="spinner"></i>正在分析工作流…</div>' : formatText(text)}</div>`;
  messages.appendChild(row);
  if (messages.children.length > 10) {
    messages.classList.add("fold-early");
    collapseMessagesButton.textContent = "展开早期";
  }
  messages.scrollTop = messages.scrollHeight;
  return row;
}

function formatText(text) {
  return String(text || "").replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char])).replace(/\n/g, "<br>");
}

function riskClass(level) {
  if (level === "高风险") return "high";
  if (level === "中风险") return "medium";
  return "low";
}

function renderResults(results) {
  if (!Array.isArray(results) || !results.length) return;
  resultCount.textContent = `${results.length} 个结果`;
  resultsContent.className = "result-list";
  resultsContent.innerHTML = results.map((item) => {
    const probability = Math.max(0, Math.min(1, Number(item.phishing_probability || 0)));
    const risk = riskClass(item.risk_level);
    return `<article class="risk-card">
      <div class="risk-card-head"><span class="url-text" title="${formatText(item.url)}">${formatText(item.url)}</span><span class="risk-pill risk-${risk}">${formatText(item.risk_level || "未知风险")}</span></div>
      <div class="probability"><strong>${(probability * 100).toFixed(1)}%</strong><div class="meter"><i class="${risk}" style="width:${probability * 100}%"></i></div></div>
      <small>${formatText(item.label || "")} · 正常概率 ${((Number(item.legitimate_probability || 0)) * 100).toFixed(1)}%</small>
    </article>`;
  }).join("");
}

function pathToUrl(path) {
  const normalized = String(path || "").replaceAll("\\", "/");
  const marker = normalized.indexOf("/outputs/");
  return marker >= 0 ? normalized.slice(marker) : `/outputs/${normalized.split("/").pop()}`;
}

function addAssets(data) {
  const assets = [
    ["report_path", "▤", "检测报告"],
    ["table_path", "▦", "结果表格"],
    ["chart_path", "◒", "风险图表"],
    ["poster_path", "▣", "安全海报"],
  ].filter(([key]) => data[key]);
  if (!assets.length) return;
  assets.forEach(([key, icon, label]) => state.assets.set(key, { path: data[key], icon, label }));
  assetsContent.innerHTML = Array.from(state.assets.values()).map(({ path, icon, label }) => {
    const url = pathToUrl(path);
    return `<div class="asset-item"><span class="asset-icon">${icon}</span><a href="${url}" target="_blank" title="${formatText(path)}">${label} · ${formatText(String(path).split(/[\\/]/).pop())}</a></div>`;
  }).join("");
  const poster = data.poster_path || data.chart_path;
  if (poster) showToast("文件已生成，可在右侧打开");
}

async function submit(prompt) {
  prompt = String(prompt || "").trim();
  if (!prompt || state.busy) return;
  state.busy = true;
  sendButton.disabled = true;
  input.value = "";
  input.style.height = "auto";
  addMessage(prompt, "user");
  const pending = addMessage("", "assistant", true);
  try {
    const response = await fetch("/api/v1/workflow", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_input: prompt, last_detection_results: state.lastResults }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "接口请求失败");
    pending.querySelector(".bubble").innerHTML = formatText(data.final_answer || data.error || "任务已完成。");
    if (data.last_detection_results) {
      state.lastResults = data.last_detection_results;
      renderResults(Array.isArray(state.lastResults) ? state.lastResults : [state.lastResults]);
    }
    addAssets(data);
  } catch (error) {
    pending.querySelector(".bubble").innerHTML = `<span style="color:#b43e40">请求失败：${formatText(error.message)}</span>`;
    showToast("请求失败，请检查服务状态");
  } finally {
    state.busy = false;
    sendButton.disabled = false;
    input.focus();
  }
}

function showToast(text) {
  toast.textContent = text;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2600);
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-prompt]");
  if (button) submit(button.dataset.prompt);
});
sendButton.addEventListener("click", () => submit(input.value));
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(input.value); }
});
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 120)}px`; });
document.querySelector("#newChat").addEventListener("click", () => window.location.reload());
collapseMessagesButton.addEventListener("click", () => {
  const folded = messages.classList.toggle("fold-early");
  collapseMessagesButton.textContent = folded ? "展开早期" : "折叠早期";
  if (!folded) messages.scrollTop = messages.scrollHeight;
});
toggleAssetsButton.addEventListener("click", () => {
  const collapsed = assetsContent.classList.toggle("collapsed");
  toggleAssetsButton.textContent = collapsed ? "展开" : "收起";
});
document.querySelector("#clearChat").addEventListener("click", () => {
  messages.innerHTML = "";
  state.lastResults = null;
  state.assets.clear();
  messages.classList.remove("fold-early");
  collapseMessagesButton.textContent = "折叠早期";
  assetsContent.classList.remove("collapsed");
  toggleAssetsButton.textContent = "收起";
  resultCount.textContent = "等待检测";
  resultsContent.className = "empty-state";
  resultsContent.innerHTML = '<div class="empty-icon">⌁</div><strong>结果将在这里呈现</strong><span>发送一个 URL 开始分析</span>';
  assetsContent.innerHTML = '<div class="asset-empty">完成检测后，可从对话或快捷任务生成文件。</div>';
});
