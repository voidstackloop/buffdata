"use strict";
const el = id => document.getElementById(id);
const csrf = document.querySelector('meta[name="csrf-token"]').content;
async function request(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...options.headers, "X-CSRF-Token": csrf}});
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Request failed");
  return body;
}
function report(error) { el("message").textContent = error.message; }
function params() { return "?project_id=" + encodeURIComponent(el("project").value); }
function button(text, handler) {
  const b = document.createElement("button"); b.textContent = text;
  b.onclick = () => Promise.resolve().then(handler).catch(report); return b;
}
async function selectRun(run) {
  el("details").textContent = JSON.stringify(await request("/api/v1/runs/" + run.id + params()), null, 2);
  el("actions").replaceChildren();
  for (const name of ["events", "compare", "verify", "cancel", "resume"]) {
    el("actions").append(button(name, async () => {
      const data = await request("/api/v1/runs/" + run.id + "/" + name + params(), {method: ["cancel", "resume"].includes(name) ? "POST" : "GET"});
      el("details").textContent = JSON.stringify(data, null, 2); await refresh();
    }));
  }
  if (run.status === "succeeded") for (const name of ["output", "rejected", "report"]) {
    const link = document.createElement("a"); link.textContent = "Download " + name;
    link.href = "/api/v1/runs/" + run.id + "/artifacts/" + name + params(); el("actions").append(link);
  }
}
async function refresh() {
  const runs = await request("/api/v1/runs" + params()); el("runs").replaceChildren();
  for (const run of runs) el("runs").append(button(run.id.slice(0, 10) + " · " + run.status + " · " + run.created_at, () => selectRun(run)));
  if (!runs.length) el("runs").textContent = "No runs yet.";
}
el("refresh").onclick = () => refresh().catch(report);
el("project").onchange = () => refresh().catch(report);
el("logout").onclick = async () => { await request("/logout", {method: "POST"}); window.location.assign("/login"); };
el("submit").onclick = async () => {
  const button = el("submit"); button.disabled = true;
  try {
    const file = el("file").files[0]; if (!file) throw new Error("Choose a dataset first");
    const configuration = JSON.parse(el("configuration").value);
    const dataset = await request("/api/v1/datasets" + params() + "&filename=" + encodeURIComponent(file.name), {method: "POST", body: file});
    const run = await request("/api/v1/runs", {method: "POST", headers: {"Content-Type": "application/json", "Idempotency-Key": crypto.randomUUID()},
      body: JSON.stringify({project_id: el("project").value, dataset_id: dataset.id, configuration})});
    el("message").textContent = "Queued " + run.id; await refresh();
  } catch (error) { report(error); } finally { button.disabled = false; }
};
(async () => {
  for (const project of await request("/api/v1/projects")) {
    const option = document.createElement("option"); option.value = project.id; option.textContent = project.id + " · " + project.role; el("project").append(option);
  }
  if (el("project").value) { await refresh(); setInterval(() => refresh().catch(report), 5000); }
})().catch(report);
