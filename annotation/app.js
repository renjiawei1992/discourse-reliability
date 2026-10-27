/* B1 annotation app.
 *
 * No framework, no network reads: the item file is a plain <script> so the
 * page works from file:// as well as over HTTP.  Everything the annotator
 * does is written to localStorage on every click, because the realistic
 * failure mode is a closed tab halfway through 300 items.
 */
(function () {
  "use strict";

  var DATA = window.B1;
  if (!DATA || !DATA.items) {
    document.body.innerHTML = "<p style='padding:40px'>缺少 items.js</p>";
    return;
  }

  var SENSES = ["TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION"];
  var KEY_LAST = "b1:last-annotator";
  var keyState = function (a) { return "b1:" + DATA.packet_hash + ":" + a; };

  var annotator = localStorage.getItem(KEY_LAST) || "";
  // ?a=B lets the server hand each annotator a personal link.
  try {
    var urlA = new URLSearchParams(location.search).get("a");
    if (urlA) annotator = urlA.trim().toUpperCase().slice(0, 8);
  } catch (e) { /* older browsers: fall back to the selector */ }
  var order = [];      // [{item, index}] in the order this annotator sees them
  var answers = {};    // id -> {s, note, ts}
  var cursor = 0;

  var $ = function (id) { return document.getElementById(id); };

  /* ---------- deterministic shuffle ---------- */

  function seedFrom(str) {
    var h = 2166136261;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function rngFrom(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffle(list, rng) {
    var a = list.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  function buildOrder() {
    // Pilot first: it exists to surface guideline problems before the real run.
    var pilot = DATA.items.filter(function (x) { return x.split === "pilot"; });
    var main = DATA.items.filter(function (x) { return x.split !== "pilot"; });
    var rng = rngFrom(seedFrom(annotator + "|" + DATA.packet_hash));
    return shuffle(pilot, rng).concat(shuffle(main, rng));
  }

  /* ---------- persistence ---------- */

  function load() {
    try {
      answers = JSON.parse(localStorage.getItem(keyState(annotator)) || "{}") || {};
    } catch (e) {
      answers = {};
    }
  }

  function save() {
    localStorage.setItem(keyState(annotator), JSON.stringify(answers));
    flash("已自动保存");
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

  // A note typed before a class is picked must not count as an answer:
  // it would inflate the progress bar and make the item look done.
  function isAnswered(id) {
    return !!(answers[id] && answers[id].s);
  }

  function answeredCount() {
    return order.filter(function (o) { return isAnswered(o.id); }).length;
  }

  function updateProgress() {
    var n = answeredCount(), total = order.length;
    $("counter").textContent = n + " / " + total;
    $("barfill").style.width = total ? (100 * n / total) + "%" : "0%";
  }

  function render() {
    var entry = order[cursor];
    if (!entry) { renderDone(); return; }
    var it = entry;
    var state = answers[it.id] || {};

    $("phase").textContent = it.split === "pilot" ? "试标" : "正式";
    $("tasktype").textContent = it.task;
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
    $("note").value = state.note || "";

    Array.prototype.forEach.call(document.querySelectorAll(".ans"), function (b) {
      b.classList.toggle("selected", state.s === b.getAttribute("data-sense"));
    });
    $("unknown").classList.toggle("selected", state.s === "UNKNOWN");

    $("prev").disabled = cursor === 0;
    $("next").disabled = cursor >= order.length - 1;
    updateProgress();
  }

  function renderDone() {
    var n = answeredCount(), total = order.length;
    $("phase").textContent = "完成";
    $("tasktype").textContent = "";
    $("idx").textContent = "";
    $("arg1").textContent = n === total
      ? "全部 " + total + " 条已完成。请点下方「导出 CSV」并把文件发给项目负责人。"
      : "当前没有更多题目。已完成 " + n + " / " + total + "。";
    $("arg2").textContent = "";
    $("connrow").classList.add("hidden");
    $("note").value = "";
    $("prev").disabled = false;
    $("next").disabled = true;
    updateProgress();
  }

  /* ---------- actions ---------- */

  function record(sense) {
    var it = order[cursor];
    if (!it) return;
    var note = $("note").value.trim();
    if (sense === "UNKNOWN" && !note) {
      $("note").focus();
      setStatus("选「无法判定」时请写一句原因。", "bad");
      return;
    }
    answers[it.id] = { s: sense, note: note, ts: new Date().toISOString() };
    save();
    render();
    advance();
  }

  function advance() {
    var next = cursor + 1;
    while (next < order.length && isAnswered(order[next].id)) next++;
    if (next >= order.length) {
      // everything after this point is answered; go to the first unanswered
      // before the cursor if any, otherwise report completion.
      for (var i = 0; i < order.length; i++) {
        if (!isAnswered(order[i].id)) { cursor = i; render(); return; }
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
    cursor = n;
    render();
  }

  /* ---------- export ---------- */

  function csvEscape(v) {
    v = (v === undefined || v === null) ? "" : String(v);
    return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }

  function buildCsv() {
    var head = ["item_id", "doc_id", "split", "task", "arg1", "arg2", "connective",
                "text", "annotator_sense_l1", "annotator_sense_l2",
                "annotator_sense_l3", "note"];
    var lines = [head.join(",")];
    order.forEach(function (it) {
      var a = answers[it.id] || {};
      lines.push([
        it.id, it.doc, it.split, it.task, it.arg1, it.arg2, it.conn, it.text,
        a.s || "", "", "", a.note || ""
      ].map(csvEscape).join(","));
    });
    return "﻿" + lines.join("\r\n") + "\r\n";
  }

  function stamp() {
    var d = new Date();
    var p = function (x) { return ("0" + x).slice(-2); };
    return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + "-" + p(d.getHours()) + p(d.getMinutes());
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
    annotator = $("annotator").value;
    if (!annotator) return;
    localStorage.setItem(KEY_LAST, annotator);
    order = buildOrder();
    load();
    var firstUnanswered = 0;
    for (var i = 0; i < order.length; i++) {
      if (!isAnswered(order[i].id)) { firstUnanswered = i; break; }
    }
    cursor = firstUnanswered;
    $("welcome").classList.add("hidden");
    $("app").classList.remove("hidden");
    $("dock").classList.remove("hidden");
    render();
  }

  $("annotator").addEventListener("change", function () {
    $("start").disabled = !this.value;
  });

  $("start").addEventListener("click", start);

  Array.prototype.forEach.call(document.querySelectorAll(".ans"), function (b) {
    b.addEventListener("click", function () {
      record(b.getAttribute("data-sense"));
    });
  });

  $("unknown").addEventListener("click", function () { record("UNKNOWN"); });

  $("note").addEventListener("input", function () {
    var it = order[cursor];
    if (!it) return;
    var prev = answers[it.id] || {};
    answers[it.id] = { s: prev.s || "", note: this.value, ts: new Date().toISOString() };
    // Debounced so typing does not spam localStorage on every keystroke.
    clearTimeout(this._t);
    var self = this;
    this._t = setTimeout(function () {
      localStorage.setItem(keyState(annotator), JSON.stringify(answers));
    }, 400);
  });

  $("prev").addEventListener("click", function () { move(-1); });
  $("next").addEventListener("click", function () { move(1); });

  $("export-csv").addEventListener("click", function () {
    if (!answeredCount()) { setStatus("还没有标注任何题目。", "bad"); return; }
    download("b1_" + annotator + "_" + stamp() + ".csv", buildCsv());
    setStatus("已导出 CSV。", "ok");
  });

  $("export-json").addEventListener("click", function () {
    download(
      "b1_" + annotator + "_" + stamp() + ".json",
      JSON.stringify({ annotator: annotator, packet: DATA.packet,
                       packet_hash: DATA.packet_hash, answers: answers }, null, 2),
      "application/json"
    );
    setStatus("已导出 JSON 备份。", "ok");
  });

  $("submit").addEventListener("click", function () {
    if (!answeredCount()) { setStatus("还没有标注任何题目。", "bad"); return; }
    setStatus("正在提交…");
    fetch("/submit?annotator=" + encodeURIComponent(annotator) +
          "&hash=" + encodeURIComponent(DATA.packet_hash), {
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
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
    if (e.key >= "1" && e.key <= "4") record(SENSES[Number(e.key) - 1]);
    else if (e.key === "u" || e.key === "U") record("UNKNOWN");
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "ArrowRight") move(1);
  });

  if (annotator) {
    var sel = $("annotator");
    if (!Array.prototype.some.call(sel.options, function (o) { return o.value === annotator; })) {
      var opt = document.createElement("option");
      opt.value = opt.textContent = annotator;
      sel.appendChild(opt);
    }
    sel.value = annotator;
    $("start").disabled = false;
  }
})();
