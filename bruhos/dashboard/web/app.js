"use strict";
const $ = (id) => document.getElementById(id);
const node = (tag, text, className) => {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
};
let episodes = [];
let selected = "all";
let lastEpisodes = "";
function renderEpisodes() {
  const filtered = episodes.filter(
    (ep) =>
      selected === "all" ||
      (selected === "completed" ? ep.success : !ep.success),
  );
  const list = $("episodes");
  list.replaceChildren();
  if (!filtered.length) {
    list.append(
      node(
        "div",
        episodes.length
          ? "No episodes match this filter."
          : "No robot activity yet. Completed skills will appear here.",
        "empty",
      ),
    );
    return;
  }
  for (const ep of filtered) {
    const detail = node("details", "", "episode");
    const summary = node("summary", "");
    const icon = node("span", ep.success ? "✳" : "↻", "episode-icon");
    icon.setAttribute("aria-hidden", "true");
    const title = node("div", "");
    title.append(
      node("div", ep.instruction, "episode-title"),
      node(
        "div",
        `${ep.body} · ${ep.n_steps} skill ${ep.n_steps === 1 ? "step" : "steps"}`,
        "episode-meta mono",
      ),
    );
    const result = node("div", "", "result");
    result.append(
      node(
        "span",
        ep.success ? "Completed" : "Stopped / failed",
        "result-status",
      ),
      node(
        "span",
        ep.verified ? "✓ Trace verified" : "Trace unverified",
        ep.verified ? "verified mono" : "unverified mono",
      ),
    );
    summary.append(icon, title, result);
    const body = node("div", "", "detail");
    const dl = node("dl", "");
    for (const [label, value] of [
      ["Episode", ep.episode_id],
      ["Brain", ep.brain],
      ["Result", ep.end_reason],
      ["Recoveries", String(ep.recoveries)],
      ["Trace root", ep.trace_root],
    ]) {
      dl.append(
        node("dt", label),
        node("dd", value, label === "Trace root" ? "hash" : ""),
      );
    }
    body.append(dl);
    detail.append(summary, body);
    list.append(detail);
  }
}
for (const button of document.querySelectorAll("[data-filter]")) {
  button.addEventListener("click", () => {
    selected = button.dataset.filter;
    for (const other of document.querySelectorAll("[data-filter]"))
      other.setAttribute("aria-pressed", String(other === button));
    renderEpisodes();
  });
}
async function update() {
  try {
    const response = await fetch("/status", {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error("Status unavailable");
    const data = await response.json();
    $("connection").classList.remove("offline");
    $("connection").lastElementChild.textContent = "Trace console connected";
    $("mode").textContent = "SAVED EPISODES · READ ONLY";
    $("version").textContent = `bruhOS ${data.version}`;
    $("total").textContent = data.episodes.length;
    $("episode-label").textContent =
      data.episodes.length === 1 ? "recent episode" : "recent episodes";
    const verifiedCount = data.episodes.filter((ep) => ep.verified).length;
    $("verified").textContent = verifiedCount;
    $("verified-label").textContent = verifiedCount === 1
      ? "trace verified" : "traces verified";
    $("refresh").textContent = data.skipped
      ? `UPDATED JUST NOW · ${data.skipped} UNREADABLE EPISODES SKIPPED`
      : "LOCAL DATA · UPDATED JUST NOW";
    const serialized = JSON.stringify(data.episodes);
    if (serialized !== lastEpisodes) {
      lastEpisodes = serialized;
      episodes = data.episodes;
      renderEpisodes();
    }
  } catch {
    $("connection").classList.add("offline");
    $("connection").lastElementChild.textContent = "Trace console disconnected";
    $("refresh").textContent = episodes.length
      ? "SHOWING LAST RECEIVED DATA"
      : "WAITING FOR TRACE CONSOLE";
    if (!lastEpisodes)
      $("episodes").replaceChildren(
        node(
          "div",
          "The trace console is unavailable. This page will reconnect automatically.",
          "empty",
        ),
      );
  }
}
async function poll() {
  await update();
  setTimeout(poll, 2000);
}
poll();
