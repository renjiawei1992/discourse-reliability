// Minimal DOM shim that exercises annotation/app.js without a browser.
//   node scripts/smoke_annotation_app.js
const store = new Map();
global.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
};
global.location = { search: "?a=A" };
global.URLSearchParams = class { constructor(s){ this.s=s||""; } get(k){
  const m = new RegExp("[?&]"+k+"=([^&]*)").exec(this.s); return m ? decodeURIComponent(m[1]) : null; } };
global.Blob = class { constructor(p){ this.p=p; global.__lastBlob = p; } };
global.URL = { createObjectURL: () => "blob:x", revokeObjectURL: () => {} };

const made = {};
function el(tag) {
  return {
    tagName: tag || "DIV", textContent: "", value: "", disabled: false, style: {},
    _cls: new Set(),
    classList: {
      add: c => made[tag + "_c"], remove: () => {}, toggle: () => {},
    },
    addEventListener(t, f) { this["on" + t] = f; },
    appendChild() {}, click() {}, remove() {}, focus() {},
    options: [{value:""},{value:"A"},{value:"B"},{value:"C"},{value:"D"}],
    getAttribute(k) { return this[k]; },
  };
}
const nodes = {};
["counter","barfill","phase","tasktype","idx","arg1","arg2","connrow","conn","text",
 "note","prev","next","saved","status","welcome","app","dock","annotator","start",
 "unknown","export-csv","export-json","submit"].forEach(id => { nodes[id] = el(); });

const ANS = ["TEMPORAL","CONTINGENCY","COMPARISON","EXPANSION"].map(s => { const e = el("BUTTON"); e["data-sense"] = s; return e; });
global.document = {
  getElementById: id => nodes[id],
  querySelectorAll: sel => (sel === ".ans" ? ANS : []),
  createElement: t => el(t),
  body: { appendChild() {} },
  addEventListener() {},
};
global.window = global;

require(require("path").join(__dirname, "..", "annotation", "items.js"));
require(require("path").join(__dirname, "..", "annotation", "app.js"));

// pick annotator, start
nodes.annotator.value = "A";
nodes.annotator.onchange.call(nodes.annotator);
console.log("start enabled:", !nodes.start.disabled, "| items:", window.B1.n);
nodes.start.onclick();
console.log("first idx:", nodes.idx.textContent);
console.log("counter:", nodes.counter.textContent);
console.log("arg1:", nodes.arg1.textContent.slice(0, 30));

// answer three items
for (let i = 0; i < 3; i++) {
  ANS[i % 4].onclick();
  console.log("after answer", i + 1, "counter:", nodes.counter.textContent, "idx:", nodes.idx.textContent);
}
// UNKNOWN without a note must be refused
const before = nodes.counter.textContent;
nodes.note.value = "";
nodes.unknown.onclick();
console.log("unknown w/o note refused:", nodes.counter.textContent === before, "| status:", nodes.status.textContent);
// UNKNOWN with a note is accepted
nodes.note.value = "两个论元读不出连贯关系";
nodes.unknown.onclick();
console.log("after unknown:", nodes.counter.textContent);
const saved = JSON.parse(store.get("b1:" + window.B1.packet_hash + ":A"));
console.log("stored:", Object.keys(saved).length, "labels:", Object.values(saved).map(x => x.s).join(","));

nodes["export-csv"].onclick();
const csv = global.__lastBlob[0];
const NL = String.fromCharCode(13, 10);
console.log("status:", nodes.status.textContent);
console.log("header:", csv.split(NL)[0]);
const rows = csv.split(NL).filter(Boolean);
console.log("rows:", rows.length - 1);
console.log("row1:", rows[1].slice(0, 90));

