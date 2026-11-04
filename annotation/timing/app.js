/* R152 timed-annotation app.
 *
 * Three differences from the B1 app, all of them in service of measuring time:
 *
 *  1. A per-item clock.  It starts when the item is painted and stops on the
 *     answer click; revisiting an item accumulates rather than restarts, so a
 *     second look is counted instead of hidden.
 *  2. A suggestion banner for conditions b/c.  It is deliberately *not*
 *     pre-selected on the four buttons: with the banner, pressing Enter is one
 *     keystroke for "accept", the same cost as clicking a button, and the four
 *     options stay visually symmetric between conditions.
 *  3. Condition-c items the triage did not flag are never rendered; they are
 *     written to the export as auto-accepted at the model label.
 *
 * No gold label is in items.js, so nothing here can leak the reference answer.
 */
(function () {
  "use strict";

  var DATA = window.R152;
  if (!DATA || !DATA.items) {
    document.body.innerHTML = "<p style='padding:40px'>缺少 items.js</p>";
    return;
  }

  var SENSES = ["TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION"];
  var ZH = { TEMPORAL: "时间", CONTINGENCY: "因果条件", COMPARISON: "对比", EXPANSION: "扩展" };
  var KEY_LAST = "r152:last-annotator";
  var KEY_SESSION = "r152:last-session";

  var params = new URLSearchParams(location.search);
  var annotator = (params.get("a") || localStorage.getItem(KEY_LAST) || "").trim();
  var session = params.get("s") || localStorage.getItem(KEY_SESSION) || "1";

  var order = [];      // item ids in the order this annotator sees them
  var cond = {};       // id -> "a" | "b" | "c" | "w"
  var answers = {};    // id -> {s, ms, visits, ts}
  var cursor = 0;
  var renderTs = 0;
  var sessionStart = 0;
  var SESS = null;     // this annotator's sessions; resolved in start()

  /* Each annotator covers half the items, so the session list depends on who
   * they are rather than on the packet alone. */
  function resolveSessions(who) {
    var slot = (DATA.slots && DATA.slots[who]) || params.get("slot") || null;
    if (slot && DATA.by_slot && DATA.by_slot[slot]) return DATA.by_slot[slot];
    return DATA.sessions || {};
  }

  function refreshSessionChoices(who) {
    var sess = resolveSessions(who);
    Array.prototype.forEach.call($("session").querySelectorAll("option"), function (o) {
      var has = !!(sess && sess[o.value]);
      o.hidden = !has;
      o.disabled = !has;
    });
    if (sess && !sess[$("session").value]) {
      var first = Object.keys(sess).sort()[0];
      if (first) $("session").value = first;
    }
  }

  var $ = function (id) { return document.getElementById(id); };
  var keyState = function () {
    return "r152:" + DATA.packet_hash + ":" + annotator + ":s" + session;
  };

  /* ---------- persistence ---------- */

  function load() {
    try {
      answers = JSON.parse(localStorage.getItem(keyState()) || "{}") || {};
    } catch (e) {
      answers = {};
    }
  }

  function persist() {
    localStorage.setItem(keyState(), JSON.stringify(answers));
  }

  var flashTimer = null;
  function flash(msg) {
    var el = $("saved");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(flashTimer);
    flashTimer = setTimeout(function () { el.classList.remove("show"); }, 1200);
  }

  /* ---------- rendering ---------- */

  function isAnswered(id) { return !!(answers[id] && answers[id].s); }

  function answeredCount() {
    return order.filter(function (id) { return isAnswered(id); }).length;
  }

  function updateProgress() {
    var n = answeredCount(), total = order.length;
    $("counter").textContent = n + " / " + total;
    $("barfill").style.width = total ? (100 * n / total) + "%" : "0%";
  }

  function render() {
    var id = order[cursor];
    if (!id) { renderDone(); return; }
    var it = DATA.items[id];
    var state = answers[id] || {};
    var c = cond[id] || "";

    $("phase").textContent = c === "w" ? "热身"
      : c === "a" ? "从零标"
      : c === "b" ? "校对建议"
      : c === "c" ? "只查标记" : "";
    $("tasktype").textContent = it.task === "implicit_relation" ? "隐式" : "显式";
    $("idx").textContent = "第 " + (cursor + 1) + " 题 · " + it.doc;

    $("arg1").textContent = it.arg1 || it.text;
    $("arg2").textContent = it.arg2 || "";

    if (it.conn) {
      $("connrow").classList.remove("hidden");
      $("conn").textContent = it.conn;
    } else {
      $("connrow").classList.add("hidden");
    }
    $("text").textContent = it.text;

    if ((c === "b" || c === "c") && it.suggestion) {
      $("suggest").classList.remove("hidden");
      $("suggest-zh").textContent = ZH[it.suggestion] || it.suggestion;
      $("suggest-en").textContent = it.suggestion;
    } else {
      $("suggest").classList.add("hidden");
    }

    Array.prototype.forEach.call(document.querySelectorAll(".ans"), function (b) {
      b.classList.toggle("selected", state.s === b.getAttribute("data-sense"));
    });
    $("unknown").classList.toggle("selected", state.s === "UNKNOWN");

    $("prev").disabled = cursor === 0;
    $("next").disabled = cursor >= order.length - 1;
    updateProgress();
    window.scrollTo(0, 0);

    // Start the clock only after the browser has painted the item; otherwise
    // the measured time would include our own render cost.
    requestAnimationFrame(function () { renderTs = performance.now(); });
  }

  function renderDone() {
    var n = answeredCount(), total = order.length;
    var mins = sessionStart ? ((performance.now() - sessionStart) / 60000).toFixed(1) : "?";
    $("suggest").classList.add("hidden");
    $("phase").textContent = "完成";
    $("tasktype").textContent = "";
    $("idx").textContent = "";
    var next = { "1": "2", "2": "3", "3": "" }[session] || "";
    if (next && !(SESS && SESS[next])) next = "";
    var trail = next ? " 稍后用 s=" + next + " 的链接打开下一个会话。" : " 会话已全部完成，谢谢。";
    $("arg1").textContent = n === total
      ? "本会话 " + total + " 题已完成，用时 " + mins + " 分钟。"
        + "请点下方「导出 CSV」或「提交」。" + trail
      : "当前没有更多题目。已完成 " + n + " / " + total + "。";
    $("arg2").textContent = "";
    $("connrow").classList.add("hidden");
    $("prev").disabled = false;
    $("next").disabled = true;
    updateProgress();
  }

  /* ---------- actions ---------- */

  function elapsed() {
    return renderTs ? performance.now() - renderTs : 0;
  }

  function record(sense) {
    var id = order[cursor];
    if (!id) return;
    var prev = answers[id] || { ms: 0, visits: 0 };
    answers[id] = {
      s: sense,
      ms: Math.round((prev.ms || 0) + elapsed()),
      visits: (prev.visits || 0) + 1,
      ts: new Date().toISOString()
    };
    persist();
    advance();
  }

  function advance() {
    var next = cursor + 1;
    while (next < order.length && isAnswered(order[next])) next++;
    if (next >= order.length) {
      for (var i = 0; i < order.length; i++) {
        if (!isAnswered(order[i])) { cursor = i; render(); return; }
      }
      cursor = order.length;
      renderDone();
      return;
    }
    cursor = next;
    render();
  }

  function move(delta) {
    var n = cursor + delta;
    if (n < 0 || n >= order.length) return;
    // Leaving an item without answering still costs time; bank it so the
    // revisit measurement stays honest.
    var id = order[cursor];
    if (id && !isAnswered(id) && renderTs) {
      var a = answers[id] || { s: "", ms: 0, visits: 0 };
      a.ms = Math.round((a.ms || 0) + elapsed());
      a.visits = (a.visits || 0) + 1;
      answers[id] = a;
      persist();
    }
    cursor = n;
    render();
  }

  /* ---------- export ---------- */

  function csvEscape(v) {
    v = (v === undefined || v === null) ? "" : String(v);
    return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }

  var HEAD = ["annotator", "session", "condition", "item_id", "doc_id", "explicit",
              "task", "arg1", "arg2", "connective", "text",
              "suggestion", "conf", "flagged", "label", "auto", "changed",
              "ms", "visits", "answered_at"];

  function row(id, a, auto) {
    var it = DATA.items[id];
    // Auto-accepted rows never pass through `cond`, which only lists the items
    // that were shown; they still belong to condition c.
    var c = auto ? "c" : (cond[id] || "");
    var label = auto ? it.suggestion : ((a && a.s) || "");
    return [
      annotator, session, c, id, it.doc, it.explicit ? "explicit" : "implicit",
      it.task, it.arg1, it.arg2, it.conn, it.text,
      it.suggestion || "", it.conf === null ? "" : it.conf, it.flagged ? 1 : 0,
      label, auto ? 1 : 0,
      (it.suggestion && label) ? (label !== it.suggestion ? 1 : 0) : "",
      auto ? 0 : ((a && a.ms) || 0),
      auto ? 0 : ((a && a.visits) || 0),
      auto ? "" : ((a && a.ts) || "")
    ].map(csvEscape).join(",");
  }

  function buildCsv() {
    var lines = [HEAD.join(",")];
    order.forEach(function (id) { lines.push(row(id, answers[id], false)); });
    // Condition c: items the triage did not flag were never shown.
    if (session === "3") {
      var auto = (SESS && SESS["3"] && SESS["3"].auto) || {};
      Object.keys(auto).sort().forEach(function (id) {
        lines.push(row(id, null, true));
      });
    }
    return "﻿" + lines.join("\r\n") + "\r\n";
  }

  function stamp() {
    var d = new Date();
    var p = function (x) { return ("0" + x).slice(-2); };
    return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + "-"
      + p(d.getHours()) + p(d.getMinutes());
  }

  function download(name, text, type) {
    var blob = new Blob([text], { type: type || "text/csv;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(url); a.remove(); }, 0);
  }

  function setStatus(msg, kind) {
    var el = $("status");
    el.textContent = msg;
    el.className = "status" + (kind ? " " + kind : "");
  }

  /* ---------- wiring ---------- */

  function start() {
    annotator = $("annotator").value.trim();
    if (!annotator) return;
    session = $("session").value;
    localStorage.setItem(KEY_LAST, annotator);
    localStorage.setItem(KEY_SESSION, session);

    SESS = resolveSessions(annotator);
    var s = SESS[session];
    if (!s) { setStatus("这个会话不存在（检查标注者编号）。", "bad"); return; }
    order = s.order.slice();
    cond = s.cond || {};
    load();

    // Resume at the first unanswered item, counting skipped ones as done only
    // if they have a label.
    var firstUnanswered = order.length;
    for (var i = 0; i < order.length; i++) {
      if (!isAnswered(order[i])) { firstUnanswered = i; break; }
    }
    cursor = firstUnanswered;

    sessionStart = performance.now();
    $("welcome").classList.add("hidden");
    $("app").classList.remove("hidden");
    $("dock").classList.remove("hidden");
    $("session").value = session;
    render();
  }

  $("annotator").addEventListener("input", function () {
    var who = this.value.trim();
    $("start").disabled = !who;
    refreshSessionChoices(who);
  });

  $("start").addEventListener("click", start);

  Array.prototype.forEach.call(document.querySelectorAll(".ans"), function (b) {
    b.addEventListener("click", function () { record(b.getAttribute("data-sense")); });
  });

  $("unknown").addEventListener("click", function () { record("UNKNOWN"); });

  $("prev").addEventListener("click", function () { move(-1); });
  $("next").addEventListener("click", function () { move(1); });

  $("export-csv").addEventListener("click", function () {
    if (!answeredCount()) { setStatus("还没有标注任何题目。", "bad"); return; }
    download("r152_s" + session + "_" + annotator + "_" + stamp() + ".csv", buildCsv());
    setStatus("已导出 CSV。", "ok");
  });

  $("export-json").addEventListener("click", function () {
    download(
      "r152_s" + session + "_" + annotator + "_" + stamp() + ".json",
      JSON.stringify({ annotator: annotator, session: session,
                       packet: DATA.packet, packet_hash: DATA.packet_hash,
                       answers: answers }, null, 2),
      "application/json"
    );
    setStatus("已导出 JSON 备份。", "ok");
  });

  $("submit").addEventListener("click", function () {
    if (!answeredCount()) { setStatus("还没有标注任何题目。", "bad"); return; }
    setStatus("正在提交…");
    fetch("/submit?annotator=" + encodeURIComponent(annotator)
          + "&session=" + encodeURIComponent(session)
          + "&hash=" + encodeURIComponent(DATA.packet_hash), {
      method: "POST",
      headers: { "Content-Type": "text/csv;charset=utf-8" },
      body: buildCsv()
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (j) {
      setStatus("已提交：" + (j.saved || "ok"), "ok");
    }).catch(function () {
      setStatus("服务器未连接，请改用「导出 CSV」并把文件发给负责人。", "bad");
    });
  });

  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT"
        || e.target.tagName === "SELECT") return;
    if (e.target.tagName === "SUMMARY") return;
    var id = order[cursor];
    var c = id ? cond[id] : "";
    var sug = id ? DATA.items[id].suggestion : "";
    if (e.key === "Enter" && sug && (c === "b" || c === "c")) {
      e.preventDefault();
      record(sug);
    } else if (e.key >= "1" && e.key <= "4") record(SENSES[Number(e.key) - 1]);
    else if (e.key === "u" || e.key === "U") record("UNKNOWN");
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "ArrowRight") move(1);
  });

  // Prefill from the URL so the organiser can hand out one link per session.
  if (annotator) {
    $("annotator").value = annotator;
    $("start").disabled = false;
  }
  refreshSessionChoices(annotator);
  if ((resolveSessions(annotator) || {})[session]) {
    $("session").value = session;
  }
})();
