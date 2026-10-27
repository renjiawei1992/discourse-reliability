import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from discourse_reliability.counterfactual import generate, no_connective, synonym_swap
from discourse_reliability.data.dimlex import load_dimlex
from discourse_reliability.data.sampling import stratified_sample
from discourse_reliability.labels import normalize_sense, sense_levels
from discourse_reliability.metrics.agreement import cohen_kappa, fleiss_kappa, macro_f1
from discourse_reliability.metrics.evaluate import evaluate_by_model
from discourse_reliability.models.mock import MockAdapter
from discourse_reliability.schema import Item
from discourse_reliability.runner import run
from discourse_reliability.triage import build_scores


def make_item(i: int, explicit: bool = True, sense: str = "COMPARISON:Contrast") -> Item:
    conn = "但是" if explicit else ""
    text = "天气很好，但是我不能出门。" if explicit else "天气很好。 我不能出门。"
    l1, l2, l3 = sense_levels(sense)
    return Item(
        id=f"t{i}",
        corpus="synthetic",
        task="explicit_sense" if explicit else "implicit_relation",
        text=text,
        connective=conn,
        arg1="天气很好",
        arg2="我不能出门",
        conn_char_start=5 if explicit else -1,
        conn_char_end=7 if explicit else -1,
        explicit=explicit,
        n_lexicon_senses=2,
        sense_l1=l1,
        sense_l2=l2,
        sense_l3=l3,
        split="train",
    )


class TestLabels(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(sense_levels("contingency:cause:result"), ("CONTINGENCY", "CONTINGENCY:Cause", "CONTINGENCY:Cause:Result"))
        self.assertEqual(sense_levels("CONTRAST"), ("COMPARISON", "COMPARISON:Contrast", "COMPARISON:Contrast"))

    def test_empty(self):
        self.assertEqual(sense_levels(""), ("", "", ""))

    def test_normalize_whitespace(self):
        self.assertEqual(normalize_sense("  TEMPORAL :  Synchronous "), "TEMPORAL:Synchronous")


class TestMetrics(unittest.TestCase):
    def test_cohen_hand_computed(self):
        a = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        b = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        self.assertAlmostEqual(cohen_kappa(a, b), 0.6, places=6)

    def test_cohen_perfect(self):
        self.assertEqual(cohen_kappa(["x"] * 5, ["x"] * 5), None)  # undefined, pe=1
        self.assertAlmostEqual(cohen_kappa(["x", "y"] * 5, ["x", "y"] * 5), 1.0)

    def test_fleiss_all_agree(self):
        table = [[2, 0], [0, 2], [1, 1]]
        # p_bar = 2/3, p_e = 1/2  ->  kappa = 1/3
        self.assertAlmostEqual(fleiss_kappa(table), 1 / 3, places=6)

    def test_macro_f1(self):
        self.assertAlmostEqual(macro_f1(["a", "b"], ["a", "b"]), 1.0)


class TestDimlex(unittest.TestCase):
    def test_parse_minimal(self):
        xml = """<?xml version='1.0' encoding='UTF-8'?>
<dimlex>
  <entry id="c1" word="但是">
    <orths><orth type="cont" canonical="1"><part type="single">但是</part></orth></orths>
    <syn><cat>cco</cat><sem><pdtb3_relation sense="COMPARISON:Contrast"/>
      <example>天气很好，但是我不能出门。</example><english_equivalent>but</english_equivalent></sem></syn>
  </entry>
</dimlex>"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lex.xml"
            p.write_text(xml, encoding="utf-8")
            items = load_dimlex(p)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].sense_l2, "COMPARISON:Contrast")
        self.assertTrue(items[0].explicit)
        self.assertFalse(items[0].paired)
        self.assertEqual(items[0].n_lexicon_senses, 1)


class TestCDTB(unittest.TestCase):
    def _pipe(self, d: Path) -> Path:
        def row(rel, conn, c2, a1, a2, sense):
            r = ["_"] * 25
            r[0], r[1], r[2], r[3] = rel, conn, conn, c2
            r[14], r[15], r[16], r[24] = (conn if rel == "Explicit" else "_"), a1, a2, sense
            return "|".join(r)

        text = "\n".join(
            [
                row("Explicit", "但是", "_", "天气很好", "我不能出门", "COMPARISON:Contrast"),
                row("Implicit", "_", "_", "下大雨了", "比赛取消了", "CONTINGENCY:Cause:Result"),
                row("EntRel", "_", "_", "甲", "乙", "_"),
            ]
        )
        p = d / "toy.pipe"
        p.write_text(text + "\n", encoding="utf-8")
        return p

    def test_autodetect_and_parse(self):
        from discourse_reliability.data.cdtb import autodetect_columns, load_cdtb

        with tempfile.TemporaryDirectory() as d:
            p = self._pipe(Path(d))
            cols = autodetect_columns([line.split("|") for line in p.read_text(encoding="utf-8").splitlines()])
            self.assertEqual(cols["relation"], 0)
            self.assertEqual(cols["sense"], 24)
            self.assertLess(cols["arg1_raw"], cols["arg2_raw"])
            items, used = load_cdtb(Path(d), columns=cols)
        self.assertEqual(len(items), 2)  # EntRel dropped
        explicit = [i for i in items if i.explicit]
        implicit = [i for i in items if not i.explicit]
        self.assertEqual(len(explicit), 1)
        self.assertEqual(len(implicit), 1)
        self.assertEqual(explicit[0].sense_l2, "COMPARISON:Contrast")
        self.assertIn("但是", explicit[0].text)
        self.assertEqual(implicit[0].task, "implicit_relation")


class TestTEDCDB(unittest.TestCase):
    def _make(self, d: Path, rows: list[dict], tokens: str) -> Path:
        (d / "Ann" / "tokens").mkdir(parents=True)
        (d / "raw" / "tokens").mkdir(parents=True)
        (d / "raw" / "tokens" / "1.txt").write_text(tokens, encoding="utf-8")
        lines = []
        for r in rows:
            f = [""] * 34
            for k, v in r.items():
                f[k] = v
            lines.append("|".join(f))
        (d / "Ann" / "tokens" / "1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return d

    def test_explicit_and_implicit(self):
        from discourse_reliability.data.tedcdb import load_tedcdb

        tokens = "天气 很 好 ， 但是 我 不 能 出 门"
        rows = [
            {0: "Explicit", 1: "9..10", 2: "Wr", 3: "Comm", 4: "Null", 5: "Null",
             8: "Comparison.Concession.Arg2-as-denier", 14: "0..7", 15: "Inh",
             20: "12..24", 21: "Inh", 31: "9..10", 32: "DEFAULT"},
            {0: "Implicit", 2: "Wr", 3: "Comm", 4: "Null", 5: "Null", 7: "因为",
             8: "Contingency.Cause.Reason", 14: "0..7", 15: "Inh",
             20: "12..24", 21: "Inh", 32: "DEFAULT"},
            {0: "EntRel", 14: "0..7", 20: "12..24"},
        ]
        with tempfile.TemporaryDirectory() as d:
            load_tedcdb(self._make(Path(d), rows, tokens), include_implicit=True)
            items = load_tedcdb(Path(d), include_implicit=True)
        self.assertEqual(len(items), 2)  # EntRel dropped
        exp, imp = items
        self.assertTrue(exp.explicit)
        self.assertEqual(exp.connective, "但是")
        self.assertEqual(exp.sense_l3, "COMPARISON:Concession:Arg2-as-denier")
        self.assertEqual(exp.text[exp.conn_char_start : exp.conn_char_end], "但是")
        self.assertIn("天气很好，但是", exp.text)
        # The implied connective is annotation-only and must not leak in.
        self.assertFalse(imp.explicit)
        self.assertEqual(imp.connective, "")
        self.assertEqual(imp.meta["implicit_connective"], "因为")
        self.assertEqual(imp.sense_l1, "CONTINGENCY")


class TestHITCDTB(unittest.TestCase):
    # Two tokenised lines, byte-identical in layout so the offsets below are
    # easy to check by hand.  Line 2 starts at char 24 once \r\n collapses.
    TOKENS = (
        "天气 很 好 ， 但是 我 不 能 出 门 。\r\n"
        "因为 下雨 ， 所以 我 不 出 门 。\r\n"
    )

    @staticmethod
    def _block(rtype, conn_span, conn, a1_span, a1, a2_span, a2, relno, label):
        return (
            "--------------------------------------------------\n"
            f"____{rtype}____\n"
            "Source:synthetic\n"
            "____Connective____\n"
            f"Span:{conn_span}\n"
            f"Content:{conn}\n"
            "____Arg1____\n"
            f"Span:{a1_span}\n"
            f"Content:{a1}\n"
            "____Arg2____\n"
            f"Span:{a2_span}\n"
            f"Content:{a2}\n"
            "____Sense____\n"
            f"RelNO:{relno}\n"
            f"Content:{label}\n"
        )

    def _make(self, d: Path) -> Path:
        g = d / "wb"
        g.mkdir(parents=True)
        # GB18030 + CRLF: exactly the on-disk shape of the real sample.
        (g / "x.txt").write_bytes(self.TOKENS.encode("gb18030"))
        blocks = [
            self._block("Explicit", "9...11", "但是", "0...7", "天气 很 好",
                        "9...23", "但是 我 不 能 出 门 。", "3-3", "让步"),
            self._block("Explicit", "24...26;32...34", "因为;所以", "24...31",
                        "因为 下雨 ，", "32...44", "所以 我 不 出 门 。", "2-1-1", "因果"),
            self._block("Implicit", "26...26", "因为", "0...7", "天气 很 好",
                        "9...23", "但是 我 不 能 出 门 。", "2-1-1", "因果"),
        ]
        (g / "x.p2").write_bytes("".join(blocks).encode("gb18030"))
        # A p1 file must be ignored by default (PDTB has no sentence-group layer).
        (g / "x.p1").write_bytes(blocks[0].encode("gb18030"))
        return d

    def test_spans_crlf_and_paired(self):
        from discourse_reliability.data.hitcdtb import load_hitcdtb

        with tempfile.TemporaryDirectory() as d:
            items = load_hitcdtb(self._make(Path(d)))
        self.assertEqual(len(items), 3)  # p1 excluded
        exp, paired, imp = items

        # CRLF normalisation: no carriage returns survive, and offsets index
        # the detokenised string exactly.
        self.assertNotIn("\r", exp.text)
        self.assertEqual(exp.text, "天气很好，但是我不能出门。")
        self.assertEqual(exp.connective, "但是")
        self.assertEqual(exp.text[exp.conn_char_start : exp.conn_char_end], "但是")
        self.assertEqual((exp.arg1, exp.arg2), ("天气很好", "但是我不能出门。"))
        self.assertEqual(exp.sense_l3, "COMPARISON:Concession")
        self.assertEqual(exp.meta["hit_relno"], "3-3")
        self.assertEqual(exp.meta["hit_l1"], "比较")

        # Paired connective: both pieces are recorded, ";" is rendered as "…".
        self.assertTrue(paired.paired)
        self.assertEqual(paired.connective, "因为…所以")
        self.assertEqual(len(paired.meta["conn_char_segments"]), 2)
        self.assertEqual(
            [paired.text[a:b] for a, b in paired.meta["conn_char_segments"]],
            ["因为", "所以"],
        )
        self.assertEqual(paired.sense_l2, "CONTINGENCY:Cause")
        self.assertEqual(paired.meta["mapping_key"], "2-1-1")

        # Implicit: the supplied connective is annotation-only.
        self.assertFalse(imp.explicit)
        self.assertEqual(imp.connective, "")
        self.assertEqual(imp.meta["implicit_connective"], "因为")
        self.assertEqual(imp.conn_char_start, -1)
        self.assertEqual(imp.task, "implicit_relation")

    def test_map_sense_longest_prefix(self):
        from discourse_reliability.data.hitcdtb import map_sense

        self.assertEqual(map_sense("4-3-2-2"), ("EXPANSION:Instantiation", "4-3-2", False))
        self.assertEqual(map_sense("2-6"), ("CONTINGENCY:Condition", "2-6", True))
        # Unknown sub-codes keep their top-level class instead of vanishing.
        self.assertEqual(map_sense("4-9"), ("EXPANSION", "4", True))
        self.assertEqual(map_sense("9-9"), ("", "", True))
        self.assertEqual(map_sense(""), ("", "", True))


class TestTriagePool(unittest.TestCase):
    """A multi-model prediction file must not be silently scored as one model."""

    def _preds(self):
        from discourse_reliability.schema import Prediction

        gold = [make_item(i, sense="COMPARISON:Contrast") for i in range(8)]
        # "strong" is right everywhere; "weak" is right half the time.
        preds = []
        for k, it in enumerate(gold):
            weak = "TEMPORAL" if k % 2 else "COMPARISON"
            preds.append(
                Prediction(
                    item_id=it.id, model="strong", regime="zero_shot",
                    label_l1="COMPARISON", confidence=1.0, samples=["COMPARISON"] * 5,
                )
            )
            preds.append(
                Prediction(
                    item_id=it.id, model="weak", regime="zero_shot",
                    label_l1=weak, confidence=0.6, samples=[weak, "COMPARISON", weak],
                )
            )
        return gold, preds

    def test_consensus_averages_models_not_samples(self):
        from discourse_reliability.schema import Prediction
        from discourse_reliability.triage import consensus_entropy

        weak = Prediction(
            item_id="x", model="weak", regime="zero_shot", label_l1="EXPANSION",
            samples=["EXPANSION", "EXPANSION", "TEMPORAL"],
        )
        strong = Prediction(
            item_id="x", model="strong", regime="zero_shot", label_l1="EXPANSION",
            samples=["EXPANSION"],
        )
        # weak: 2/3 vs 1/3; strong: 1.0.  Mean = (5/6, 1/6).
        expected = -(5 / 6) * math.log(5 / 6) - (1 / 6) * math.log(1 / 6)
        got = consensus_entropy({"x": {"weak": weak, "strong": strong}}, level="l1")["x"]
        self.assertAlmostEqual(got, expected, places=6)

    def test_base_model_selects_scored_prediction(self):
        from discourse_reliability.triage import build_scores

        gold, preds = self._preds()
        # The oracle score sums to the number of items the scored model got
        # wrong, so it names which prediction file was actually triaged.
        scores_strong, _ = build_scores(gold, preds, {}, n_folds=2, level="l1", base_model="strong")
        scores_weak, _ = build_scores(gold, preds, {}, n_folds=2, level="l1", base_model="weak")
        self.assertEqual(sum(scores_strong["oracle"].values()), 0.0)
        self.assertEqual(sum(scores_weak["oracle"].values()), 4.0)


class TestL1Protocol(unittest.TestCase):
    def test_prompt_switches_to_four_classes(self):
        from discourse_reliability.prompts import build_messages

        item = make_item(1, explicit=True, sense="COMPARISON:Contrast")
        msg = build_messages(item, level="l1")[1]["content"]
        self.assertIn("TEMPORAL / CONTINGENCY / COMPARISON / EXPANSION", msg)
        self.assertIn("大类", msg)
        # The L3 template would tell the model to aim for level 2/3 labels.
        self.assertNotIn("请尽量给出二级或三级义项", msg)
        self.assertNotIn("CONTINGENCY:Cause:Result", msg)

    def test_doc_split_never_crosses_documents(self):
        from discourse_reliability.data.sampling import assign_splits

        items = []
        for d in range(20):
            for j in range(5):
                it = make_item(d * 10 + j)
                it.doc_id = f"doc{d:02d}"
                items.append(it)
        assign_splits(items, seed=7, group_by="doc_id")
        by_doc: dict[str, set[str]] = {}
        for it in items:
            by_doc.setdefault(it.doc_id, set()).add(it.split)
        self.assertTrue(all(len(v) == 1 for v in by_doc.values()), by_doc)
        self.assertEqual(len({it.split for it in items}), 3)

    def test_triage_uses_requested_level(self):
        from discourse_reliability.schema import Prediction
        from discourse_reliability.triage import build_scores

        gold = [make_item(i, sense="COMPARISON:Contrast") for i in range(12)]
        # Predictions agree at L1 but disagree at L3.
        preds = [
            Prediction(
                item_id=it.id,
                model="m",
                regime="zero_shot",
                label_l1="COMPARISON",
                label_l2="COMPARISON:Contrast",
                label_l3="COMPARISON:Contrast",
                confidence=1.0,
            )
            for it in gold
        ]
        _, auc_l1 = build_scores(gold, preds, {}, n_folds=2, level="l1")
        # No errors at L1 -> error AUC is undefined (nan), not a spurious signal.
        self.assertNotEqual(auc_l1["oracle"], 1.0)


class TestSampling(unittest.TestCase):
    def test_deterministic_and_balanced(self):
        items = [make_item(i, explicit=(i % 3 == 0)) for i in range(90)]
        a = stratified_sample(items, n=30, seed=1, explicit_fraction=1 / 3)
        b = stratified_sample(items, n=30, seed=1, explicit_fraction=1 / 3)
        self.assertEqual([x.id for x in a], [x.id for x in b])
        self.assertEqual(len(a), 30)
        n_exp = sum(x.explicit for x in a)
        self.assertTrue(8 <= n_exp <= 12, n_exp)


class TestCounterfactual(unittest.TestCase):
    def test_delete(self):
        v = no_connective(make_item(1))
        self.assertIn("____", v.text)
        self.assertNotIn("但是", v.text)
        self.assertEqual(v.gold, "COMPARISON:Contrast")

    def test_synonym(self):
        v = synonym_swap(make_item(1))
        self.assertIsNotNone(v)
        self.assertNotEqual(v.connective, "但是")
        self.assertIn(v.connective, v.text)

    def test_generate_counts(self):
        items = [make_item(1), make_item(2, explicit=False)]
        vs = generate(items, only_explicit=True)
        self.assertEqual(len(vs), 3)  # delete + synonym + reorder


class TestEndToEnd(unittest.TestCase):
    def test_mock_pipeline(self):
        senses = ["COMPARISON:Contrast", "CONTINGENCY:Cause:Result", "EXPANSION:Conjunction"]
        items = [make_item(i, explicit=(i % 2 == 0), sense=senses[i % 3]) for i in range(30)]
        adapter = MockAdapter(error_rate=0.2, seed=1)
        preds = run(items, adapter, regime="zero_shot", n_samples=3, workers=2)
        self.assertEqual(len(preds), 30)
        self.assertTrue(all(p.label_l3 for p in preds))

        report = evaluate_by_model(items, preds, n_boot=50)
        key = next(iter(report))
        self.assertIn("l2", report[key]["levels"])
        self.assertGreater(report[key]["levels"]["l2"]["n"], 0)

        scores, auc = build_scores(items, preds, {it.id: 1.0 for it in items}, n_folds=3)
        self.assertIn("hybrid", scores)
        self.assertEqual(len(scores["random"]), 30)

    def test_prior_separation(self):
        from discourse_reliability.baselines.lexeme_prior import prior_predictions

        items = [make_item(i) for i in range(20)]
        preds = prior_predictions(items, k=4)
        self.assertEqual(len(preds), 20)
        self.assertTrue(all(p.label_l3 for p in preds))


class TestAnnotationAgreement(unittest.TestCase):
    def _write_csv(self, path, rows, column="annotator_sense_l1"):
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", column])
            for item_id, label in rows:
                w.writerow([item_id, label])

    def test_undecided_excluded_and_ci_reported(self):
        from discourse_reliability.annotation import agreement

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            items = tmp / "items.jsonl"
            rows, ids = [], []
            for k in range(12):
                it = make_item(k, sense="COMPARISON:Contrast")
                it.doc_id = f"doc{k % 3}"
                it.explicit = k % 2 == 0
                ids.append(it.id)
                rows.append(it.to_dict())
            items.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
            )

            a = tmp / "a.csv"
            b = tmp / "b.csv"
            self._write_csv(a, [(ids[k], "COMPARISON") for k in range(12)])
            labels = ["COMPARISON"] * 12
            labels[0] = "TEMPORAL"   # genuine disagreement
            labels[1] = "UNKNOWN"    # one annotator gave up
            self._write_csv(b, [(ids[k], labels[k]) for k in range(12)])

            rep = agreement(a, b, column="annotator_sense_l1", items_path=items, n_boot=100)
            # UNKNOWN is not a sense: it is reported, not counted as agreement.
            self.assertEqual(rep["undecided"], 1)
            self.assertEqual(rep["n"], 11)
            self.assertIn("kappa_ci95_doc_cluster", rep["by_level"]["l1"])
            self.assertIn("explicit", rep["by_stratum"])
            # 6 even-indexed items are explicit; the UNKNOWN one (k=1) is implicit.
            self.assertEqual(rep["by_stratum"]["explicit"]["n"], 6)
            self.assertEqual(rep["by_stratum"]["implicit"]["n"], 5)


class TestValidityMetrics(unittest.TestCase):
    def test_js_divergence_bounds(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "validity_analysis", Path(__file__).resolve().parents[1] / "scripts" / "validity_analysis.py")
        va = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(va)

        same = {"a": 3, "b": 1}
        self.assertAlmostEqual(va.js_divergence(same, same), 0.0, places=9)
        disjoint = va.js_divergence({"a": 1}, {"b": 1})
        self.assertAlmostEqual(disjoint, 1.0, places=6)   # log base 2

    def test_auc_binary_detects_planted_order(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "validity_analysis", Path(__file__).resolve().parents[1] / "scripts" / "validity_analysis.py")
        va = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(va)

        score = [1, 0, 1, 0, 1, 0]
        self.assertAlmostEqual(va.auc_binary(score, score), 1.0)
        self.assertAlmostEqual(va.auc_binary(score, [0, 1, 0, 1, 0, 1]), 0.0)
        self.assertAlmostEqual(va.auc_binary([1, 0], [1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
