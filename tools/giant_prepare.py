#!/usr/bin/env python3
"""Build the giant-model training corpus from 16 real domains.

RCA fixes baked in (the point of this corpus):
  Failure 2 (identical per-row instructions are hub pollution at learn)
      -> instructions are EMPTY on every row of every domain.
  Failure 3 (natural labels tokenize into shared hubs: "card", "true"...)
      -> globally-unique opaque anchors per class (te00, ga00, gu00, c00..,
         sm00, cl000...); no anchor token is shared across domains.
  Failure 1 (sequential multi-domain training makes late domains weak)
      -> domains are ROUND-ROBIN INTERLEAVED in the train file, so every
         domain gets lessons in every region of the stream; 3-epoch
         consolidation then strengthens rather than drowns.

Also emits: per-script calibration files, per-domain hidden eval files
(router + class questions per row), a handcrafted probe file, schemas.json
(anchors, class names, criteria) and MANIFEST.json (counts + SHA-256).

Hidden files carry "hidden" in their names -> the M1 firewall refuses them
for learn/calibrate. Data policy: every state string is verbatim dataset
text (truncation is the only edit). Criteria for pre-existing domains are
preserved verbatim (only keys remapped to anchors); criteria for the new HF
domains are mined from TRAIN rows only (log-ratio, top-20).
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bank77_prepare import mine_criteria  # noqa: E402

GIANT = ROOT / "data" / "giant"
HF_RAW = ROOT / "data" / "hf_raw"
BIG = ROOT / "data" / "big"
SEED = 77

# domain -> question key + anchor prefix + class-name resolution
DOMAINS = {
    "tickets_en": {"qkey": "department", "prefix": "te", "source": "big",
                   "router": "support billing sales technical"},
    "tickets_bn": {"qkey": "department", "prefix": "tb", "source": "big",
                   "router": "support tickets bengali"},
    "tickets_hi": {"qkey": "department", "prefix": "th", "source": "big",
                   "router": "support tickets hindi"},
    "tickets_ru": {"qkey": "department", "prefix": "tr", "source": "big",
                   "router": "support tickets russian"},
    "game":       {"qkey": "action", "prefix": "ga", "source": "big",
                   "router": "game commands actions"},
    "guard":      {"qkey": "irreversible", "prefix": "gu", "source": "big",
                   "router": "safety destructive irreversible",
                   "keep_schema": True},  # noul: instructions are the probe carrier
    "bank77":     {"qkey": "intent", "prefix": "c", "source": "bank77",
                   "router": "banking cards payments"},
    "sms":        {"qkey": "intent", "prefix": "sm", "source": "sms",
                   "router": "text messages spam"},
    "clinc":      {"qkey": "intent", "prefix": "cl", "source": "hf",
                   "router": "assistant requests weather"},
    "enron":      {"qkey": "spam", "prefix": "en", "source": "hf",
                   "router": "email inbox messages"},
    "emotion":    {"qkey": "emotion", "prefix": "em", "source": "hf",
                   "router": "feelings emotions reactions"},
    "hate":       {"qkey": "speech", "prefix": "ha", "source": "hf",
                   "router": "social offensive posts"},
    "agnews":     {"qkey": "topic", "prefix": "ag", "source": "hf",
                   "router": "news headlines sports"},
    "polarity":   {"qkey": "sentiment", "prefix": "po", "source": "hf",
                   "router": "product reviews opinions"},
    "dbpedia":    {"qkey": "entity", "prefix": "db", "source": "hf",
                   "router": "encyclopedia entities topics"},
    "boolq":      {"qkey": "truth", "prefix": "bo", "source": "hf",
                   "router": "boolean question passage"},
}

CLASS_NAMES = {
    "agnews": {"0": "world", "1": "sports", "2": "business", "3": "scitech"},
    "emotion": {"0": "sadness", "1": "joy", "2": "love", "3": "anger",
                "4": "fear", "5": "surprise"},
    "polarity": {"0": "negative", "1": "positive"},
    "hate": {"0": "hate", "1": "offensive", "2": "neither"},
    "enron": {"0": "ham", "1": "spam"},
    "boolq": {"False": "false", "True": "true"},
    "dbpedia": {str(i): n for i, n in enumerate(
        ["company", "educational_institution", "artist", "athlete",
         "office_holder", "mean_of_transportation", "building",
         "natural_place", "village", "animal", "plant", "album", "film",
         "written_work"])},
}

TEXT_CAPS = {"enron": 400, "hate": 280, "agnews": 250, "polarity": 350,
             "dbpedia": 250, "boolq": 350}
TRAIN_CAPS = {"clinc": 8000, "enron": 4000, "emotion": 4000, "hate": 4078,
              "agnews": 4000, "polarity": 4000, "dbpedia": 4000, "boolq": 4000}
HIDDEN_CAPS = {"clinc": 4500, "enron": 2000, "emotion": 2000, "hate": 1500,
               "agnews": 2000, "polarity": 2000, "dbpedia": 2000, "boolq": 1500}
CAL_CAPS = {"clinc": 3000, "enron": 800, "emotion": 700, "hate": 400,
            "agnews": 400, "polarity": 400, "dbpedia": 400, "boolq": 500}
IMBALANCE_TRAIN_CAP = {"hate": {"1": 2500}}   # offensive: real-world 78% share


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in open(p) if l.strip()]


def dump_jsonl(p: Path, rows: list) -> None:
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hf_load(dom: str, tag: str) -> list:
    p = HF_RAW / f"{dom}_{tag}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def hf_state(dom: str, r: dict) -> str:
    t = " ".join(str(r["text"]).split())
    cap = TEXT_CAPS.get(dom, 0)
    return t[:cap] if cap else t


def stratified_take(rows: list, n: int, rng: random.Random) -> list:
    """Deterministic stratified sample of n rows (as balanced as possible)."""
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        for v in by.values():
            if v:
                out.append(v.pop())
                if len(out) >= n:
                    break
        i += 1
    return out


def carve(dom: str, rows: list) -> dict:
    """For domains without official cal/hidden: carve hidden + cal out."""
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    hidden_n, cal_n = HIDDEN_CAPS.get(dom, 0), CAL_CAPS.get(dom, 0)
    hidden, cal = [], []
    for cls, v in by.items():
        h = min(len(v), max(1, round(hidden_n * len(v) / len(rows))))
        hidden.extend(v[:h])
        c = min(len(v) - h, max(1, round(cal_n * len(v) / len(rows))))
        cal.extend(v[h:h + c])
        del v[:h + c]
    train = [r for v in by.values() for r in v]
    return {"train": train, "cal": cal, "hidden": hidden}


# --------------------------------------------------------------------------
# domain loading -> {"train": [...], "cal": [...], "hidden": [...]} with
# {"text", "label"} rows; labels are the DOMAIN's natural class names.

def norm_big(dom: str, split: str) -> list:
    out = []
    for r in load_jsonl(BIG / f"{dom}_{split}.jsonl"):
        qk = next(iter(r["questions"]))
        if r["state"].strip() and r["labels"].get(qk):
            out.append({"text": r["state"], "label": r["labels"][qk],
                        "row": r})
    return out


def norm_bank77(split: str) -> list:
    fn = {"train": "bank77_train.jsonl", "cal": "bank77_cal_bare.jsonl",
          "hidden": "bank77_hidden.jsonl"}[split]
    out = []
    for r in load_jsonl(ROOT / "data" / fn):
        if r["state"].strip():
            out.append({"text": r["state"], "label": r["labels"]["intent"],
                        "row": r})
    return out


def norm_sms(split: str) -> list:
    out = []
    for r in load_jsonl(ROOT / "data" / f"sms_{split}.jsonl"):
        if r["state"].strip():
            out.append({"text": r["state"], "label": r["labels"]["intent"],
                        "row": r})
    return out


def norm_hf_split(dom: str, split: str) -> list:
    out = []
    for r in hf_load(dom, split):
        if r["text"].strip() and r["label"] is not None:
            out.append({"text": hf_state(dom, r), "label": str(r["label"])})
    return out


def _ids(rows: list) -> set:
    return {(r["text"], r["label"]) for r in rows}


def _drop(rows: list, ids: set) -> list:
    return [r for r in rows if (r["text"], r["label"]) not in ids]


def load_domain(dom: str) -> dict:
    spec = DOMAINS[dom]
    src = spec["source"]
    if src == "big":
        return {s: norm_big(dom, s) for s in ("train", "cal", "hidden")}
    if src == "bank77":
        return {s: norm_bank77(s) for s in ("train", "cal", "hidden")}
    if src == "sms":
        return {s: norm_sms(s) for s in ("train", "cal", "hidden")}
    pools = {s: norm_hf_split(dom, s) for s in ("train", "cal", "hidden")}
    rng = random.Random(SEED)
    if dom == "hate":                      # single official split -> carve
        pools = carve(dom, pools["train"])
    if not pools["cal"]:                   # official train+test only
        pools["cal"] = stratified_take(pools["train"], CAL_CAPS[dom], rng)
        pools["train"] = _drop(pools["train"], _ids(pools["cal"]))
    if dom == "boolq":                     # one validation pool -> cal+hidden
        pool = pools["cal"]
        pools["hidden"] = stratified_take(pool, HIDDEN_CAPS[dom], rng)
        pools["cal"] = stratified_take(_drop(pool, _ids(pools["hidden"])),
                                       CAL_CAPS[dom], rng)
    # global disjointness: cal/hidden may never overlap each other or train
    pools["hidden"] = _drop(pools["hidden"], _ids(pools["cal"]))
    pools["hidden"] = _drop(pools["hidden"], _ids(pools["train"]))
    pools["cal"] = _drop(pools["cal"], _ids(pools["train"]))
    if dom in TRAIN_CAPS:
        pool = pools["train"]
        for lbl, cap in IMBALANCE_TRAIN_CAP.get(dom, {}).items():
            group = [r for r in pool if r["label"] == lbl]
            rest = [r for r in pool if r["label"] != lbl]
            pool = stratified_take(group, cap, rng) + rest
            rng.shuffle(pool)
        if len(pool) > TRAIN_CAPS[dom]:
            pool = stratified_take(pool, TRAIN_CAPS[dom], rng)
        pools["train"] = pool
    if len(pools["hidden"]) > HIDDEN_CAPS.get(dom, 10**9):
        pools["hidden"] = stratified_take(pools["hidden"], HIDDEN_CAPS[dom], rng)
    if len(pools["cal"]) > CAL_CAPS.get(dom, 10**9):
        pools["cal"] = stratified_take(pools["cal"], CAL_CAPS[dom], rng)
    return pools


def alloc_anchors(dom: str, classes: list) -> dict:
    """Natural class name (or bank77 c## id) -> unique opaque anchor."""
    spec = DOMAINS[dom]
    if dom == "bank77":
        return {c: c for c in classes}          # already opaque (c00..c76)
    width = 3 if len(classes) > 99 else 2
    return {c: f"{spec['prefix']}{i:0{width}d}" for i, c in enumerate(classes)}


def base_criteria(dom: str, train: list, anchor_of: dict) -> dict:
    """anchor -> criteria text. Existing domains keep their criteria verbatim
    (keys remapped); new HF domains mine top-20 from TRAIN rows only."""
    spec = DOMAINS[dom]
    src = spec["source"]
    if src in ("big", "bank77", "sms"):
        c0 = train[0]["row"]["questions"][spec["qkey"]]["criteria"]
        if dom == "bank77":
            return {k: v for k, v in c0.items()}
        return {anchor_of[nat]: txt for nat, txt in c0.items()
                if nat in anchor_of}
    by = defaultdict(list)
    for r in train:
        by[r["label"]].append(r["text"])
    mined = mine_criteria(dict(by), 10, 2)
    return {anchor_of[c]: mined.get(c, "") for c in by}


# --------------------------------------------------------------------------
# Handcrafted probes (written here, NOT copied from any dataset). For clinc
# the expected intent resolves at runtime by substring hint against the real
# 151 intent names. expected=None -> OOD probe, should defer.

PROBES = [
    ("tickets_en", "I was charged twice on my last invoice for the same seat, please refund the duplicate.", "billing"),
    ("tickets_en", "Our whole team gets a login error since this morning, the page just spins.", "technical"),
    ("tickets_en", "We want to move 40 seats from the starter plan to the annual business plan.", "sales"),
    ("tickets_bn", "আমার কার্ড থেকে দুইবার টাকা কেটেছে, অনুগ্রহ করে ফেরত দিন।", "billing"),
    ("tickets_ru", "Списали деньги дважды за одну подписку, верните пожалуйста.", "billing"),
    ("game", "Two skeletons flank the corridor ahead and my health is low; I retreat toward the torchlit hall.", "flee"),
    ("game", "The wall ahead is cracked and my pickaxe is fresh.", "dig_in"),
    ("game", "One zombie blocks the only doorway; my blade is sharp and I have room to swing.", "fight"),
    ("guard", "Delete every user account and all audit logs right now, we do not need them anymore.", "true"),
    ("guard", "Send the monthly summary email to the team alias.", "false"),
    ("bank77", "Why was I charged an extra fee on my card payment?", "card_payment_fee_charged"),
    ("bank77", "I transferred money to a friend two days ago and it has not arrived yet.", "transfer_not_received_by_recipient"),
    ("bank77", "Please cancel the transfer I made this morning.", "cancel_transfer"),
    ("sms", "Congratulations! You won a £1000 prize. Call 09058094518 now to claim!", "spam"),
    ("sms", "Hey, running 10 min late, see you at the usual spot.", "ham"),
    ("clinc", "does it look cloudy outside today", ["weather"]),
    ("clinc", "book me a table for two tonight", ["restaurant"]),
    ("clinc", "wake me up at six tomorrow morning", ["alarm"]),
    ("clinc", "how do you say thank you in japanese", ["translate"]),
    ("enron", "Team, attached are the Q3 numbers for the board meeting, send comments before Friday.", "ham"),
    ("enron", "URGENT NOTICE: your mailbox will be deactivated, verify your password at the link now!", "spam"),
    ("emotion", "I finally got the job offer I have been dreaming about, I am over the moon!", "joy"),
    ("emotion", "She left without saying goodbye and I keep staring at my phone.", "sadness"),
    ("emotion", "That sudden noise in the dark hallway made my heart race.", "fear"),
    ("hate", "People like you ruin everything, you should not be allowed here.", "offensive"),
    ("hate", "The team played terribly last night and the coach should be questioned.", "neither"),
    ("hate", "I hate mondays, everything is awful today.", "neither"),
    ("agnews", "Oil prices slide as OPEC members signal output increase", "business"),
    ("agnews", "The quarterback threw five touchdowns in the season opener", "sports"),
    ("agnews", "UN climate summit ends with a new emissions pledge", "world"),
    ("polarity", "The headphones broke after two days and support ignored my emails.", "negative"),
    ("polarity", "Exactly what I hoped for: sturdy, elegant and worth every cent.", "positive"),
    ("dbpedia", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.", "building"),
    ("dbpedia", "The Amazon river dolphin is a freshwater cetacean native to South America.", "animal"),
    ("dbpedia", "Toyota is a Japanese multinational automotive manufacturer headquartered in Aichi.", "company"),
    ("boolq", "is the zugspitze the highest mountain in germany? The Zugspitze is Germany's highest peak at 2,962 metres.", "true"),
    ("boolq", "did the roman empire fall in 476 bc? Romulus Augustulus was deposed in AD 476.", "false"),
]

OOD_PROBES = [
    "zzqvv plork wumfus xanquu blorp",
    "fragile dashboard vector quantum tulip echo forty-two",
    "seventeen purple o'clock beneath soon the",
    "qq zz pp xx vv kk jj ww ff dd",
]


def clinc_intent_names() -> list:
    """Ordered intent-name list (int id -> name), cached from first-rows."""
    cache = ROOT / "data" / "hf_cache" / "clinc_names.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import urllib.request
    url = ("https://datasets-server.huggingface.co/first-rows"
           "?dataset=clinc%2Fclinc_oos&config=plus&split=validation")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode())
    names = next(f["type"]["names"] for f in data["features"]
                 if f["name"] == "intent")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(names))
    return names


def main() -> None:
    GIANT.mkdir(parents=True, exist_ok=True)
    names = clinc_intent_names()
    schemas, pools_by_dom, anchor_maps = {}, {}, {}
    for dom in sorted(DOMAINS):
        pools = load_domain(dom)
        spec = DOMAINS[dom]
        if spec.get("keep_schema"):
            # noul domain: keep the proven multi-question schema verbatim
            tmpl = pools["train"][0]["row"]["questions"]
            schemas[dom] = {"qkey": spec["qkey"], "source": spec["source"],
                            "type": "noul_multi", "classes": {},
                            "criteria": {}, "router_anchor": None,
                            "template_questions": tmpl}
            anchor_maps[dom] = {}
            pools_by_dom[dom] = pools
            print(f"{dom:11s} {spec['source']:6s} train={len(pools['train']):6d} "
                  f"cal={len(pools['cal']):5d} hidden={len(pools['hidden']):5d} "
                  f"classes=noul(x2)+scope")
            continue
        classes = sorted({r["label"] for r in pools["train"] + pools["cal"]
                          + pools["hidden"]})
        anchor_of = alloc_anchors(dom, classes)
        if dom == "bank77":
            cat = json.load(open(ROOT / "data" / "bank77_categories.json"))
            inv_lbl = {v: k for k, v in cat["label_of"].items()}
            name_of = {c: inv_lbl[c] for c in classes}
        elif dom == "clinc":
            name_of = {c: names[int(c)] for c in classes}
        elif spec["source"] == "hf":
            name_of = {c: CLASS_NAMES[dom].get(c, c) for c in classes}
        else:
            name_of = {c: c for c in classes}
        crit = base_criteria(dom, pools["train"], anchor_of)
        anchor_maps[dom] = anchor_of
        schemas[dom] = {"qkey": spec["qkey"], "source": spec["source"],
                        "classes": {anchor_of[c]: name_of[c] for c in classes},
                        "criteria": crit, "router_anchor": None}
        pools_by_dom[dom] = pools
        print(f"{dom:11s} {spec['source']:6s} train={len(pools['train']):6d} "
              f"cal={len(pools['cal']):5d} hidden={len(pools['hidden']):5d} "
              f"classes={len(classes)}")

    doms = sorted(DOMAINS)
    rt_of = {d: f"rt{i:02d}" for i, d in enumerate(doms)}
    for d in doms:
        schemas[d]["router_anchor"] = rt_of[d]
    router_q = {"type": "choice", "instructions": "",
                "criteria": {rt_of[d]: DOMAINS[d]["router"] for d in doms}}
    schemas["_router"] = {"criteria": router_q["criteria"]}

    seen = {}
    for d in doms:
        for a in schemas[d]["classes"]:
            assert a not in seen, f"anchor collision {a}: {d} vs {seen[a]}"
            seen[a] = d
    print(f"anchors: {len(seen)} unique across {len(doms)} domains")

    def class_q_of(dom: str) -> dict:
        return {"type": "choice", "instructions": "",
                "criteria": schemas[dom]["criteria"]}

    # emit + write ----------------------------------------------------------
    cal_by_script, hidden_out, manifest_rows = {}, {}, []
    SCRIPT_OF = {"tickets_bn": "bengali", "tickets_hi": "devanagari",
                 "tickets_ru": "cyrillic"}
    queues = {d: [] for d in doms}          # per-domain train queues
    for dom in doms:
        pools = pools_by_dom[dom]
        rt = rt_of[dom]
        if DOMAINS[dom].get("keep_schema"):
            tmpl = schemas[dom]["template_questions"]
            for split, rows in pools.items():
                conv = [{"state": r["row"]["state"],
                         "questions": {"domain": router_q,
                                       **r["row"]["questions"]},
                         "labels": {"domain": rt, **r["row"]["labels"]}}
                        for r in rows]
                if split == "train":
                    queues[dom].extend(conv)
                elif split == "cal":
                    cal_by_script.setdefault("latin", []).extend(conv)
                else:
                    hidden_out[dom] = conv
            manifest_rows.append(
                {"domain": dom, "source": "big", "schema": "noul_multi",
                 "train": len(pools["train"]), "cal": len(pools["cal"]),
                 "hidden": len(pools["hidden"]), "classes": 2})
            continue
        qkey = DOMAINS[dom]["qkey"]
        cq = class_q_of(dom)
        amap = anchor_maps[dom]
        for split, rows in pools.items():
            conv = [{"state": r["text"], "questions": {"domain": router_q,
                                                       qkey: cq},
                     "labels": {"domain": rt, qkey: amap[r["label"]]}}
                    for r in rows]
            if split == "train":
                queues[dom].extend(conv)
            elif split == "cal":
                sc = SCRIPT_OF.get(dom, "latin")
                cal_by_script.setdefault(sc, []).extend(conv)
            else:
                hidden_out[dom] = conv
        manifest_rows.append({"domain": dom, "source": schemas[dom]["source"],
                              "train": len(pools["train"]),
                              "cal": len(pools["cal"]),
                              "hidden": len(pools["hidden"]),
                              "classes": len(schemas[dom]["classes"])})

    # round-robin interleave across domains (Failure-1 fix)
    interleaved, exhausted = [], set()
    while len(exhausted) < len(doms):
        for d in doms:
            if d in exhausted:
                continue
            if not queues[d]:
                exhausted.add(d)
                continue
            interleaved.append(queues[d].pop(0))
    dump_jsonl(GIANT / "train.jsonl", interleaved)

    for sc, rows in sorted(cal_by_script.items()):
        dump_jsonl(GIANT / f"cal_{sc}.jsonl", rows)
    for dom, rows in sorted(hidden_out.items()):
        dump_jsonl(GIANT / f"{dom}_hidden.jsonl", rows)

    # handcrafted probes ----------------------------------------------------
    probes = []
    pid = 0
    for dom, state, expected in PROBES:
        spec = DOMAINS[dom]
        qkey = spec["qkey"]
        if spec.get("keep_schema"):
            questions = {"domain": router_q,
                         **schemas[dom]["template_questions"]}
        else:
            questions = {"domain": router_q, qkey: class_q_of(dom)}
        if dom == "clinc":
            hits = [c for c in schemas[dom]["classes"]
                    if any(h in schemas[dom]["classes"][c] for h in expected)]
            if not hits:
                continue
            anchor = hits[0]
        elif spec.get("keep_schema"):
            anchor = expected          # noul: literal "true"/"false"
        else:
            inv = {v: k for k, v in schemas[dom]["classes"].items()}
            anchor = inv[expected]
        probes.append({"id": f"p{pid:02d}", "domain": dom, "state": state,
                       "expected_router": rt_of[dom], "expected_class": anchor,
                       "questions": questions})
        pid += 1
    for state in OOD_PROBES:
        dom = "tickets_en"                 # most familiar question schema
        probes.append({"id": f"p{pid:02d}", "domain": dom, "state": state,
                       "expected_router": rt_of[dom], "expected_class": None,
                       "questions": {"domain": router_q,
                                     "department": class_q_of(dom)}})
        pid += 1
    dump_jsonl(GIANT / "probes.jsonl", probes)

    # schemas + manifest ----------------------------------------------------
    (GIANT / "schemas.json").write_text(json.dumps(
        {"domains": schemas, "router": rt_of}, ensure_ascii=False, indent=1))
    files = {}
    for f in sorted(GIANT.glob("*.jsonl")) + [GIANT / "schemas.json"]:
        files[f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    manifest = {
        "seed": SEED,
        "recipe": {
            "instructions": "empty on every row (Failure-2 fix)",
            "anchors": "globally-unique opaque per-class anchors (Failure-3 fix)",
            "train_order": "round-robin domain interleave (Failure-1 fix)",
            "epochs": 3,
            "criteria": "existing domains verbatim (keys remapped); new HF domains mined top-20 from train only",
        },
        "sources": {
            "big": "data/big/* synthetic corpora (SHA-256 MANIFEST.json)",
            "bank77": "PolyAI banking77 official splits",
            "sms": "UCI SMS Spam Collection",
            "hf": ["clinc/clinc_oos", "SetFit/enron_spam", "dair-ai/emotion",
                   "tdavidson/hate_speech_offensive", "fancyzhx/ag_news",
                   "fancyzhx/amazon_polarity", "fancyzhx/dbpedia_14",
                   "google/boolq"],
        },
        "domains": manifest_rows,
        "train_rows": len(interleaved),
        "lessons_per_epoch": 2 * len(interleaved),
        "files": files,
    }
    (GIANT / "MANIFEST.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=1))
    print(f"train rows {len(interleaved)} ({2*len(interleaved)} lessons/epoch)")
    print(f"probes {len(probes)} ({len(OOD_PROBES)} OOD)")
    print(f"cal files: { {k: len(v) for k, v in cal_by_script.items()} }")


if __name__ == "__main__":
    main()
