#!/usr/bin/env python3
# ============================================================================
#  zeroshot_suite.py — v3.2 final zero-shot probe suite (manual-CLI parity).
#
#  Four families, none of them trained in the target fabrics:
#    EMAIL_SPAM  — spam/ham over model-tickets-big-latin (support tickets only;
#                  no spam lessons ever) — plain fabric vs its semantic rebuild
#    SMS_SPAM    — same fabrics, SMS-length texts
#    SNAKE       — 8 archived scenarios (docs/ZEROSHOT.md Part 1) over
#                  model-giant-latin (game domain trained, zero Snake lessons)
#    TICTACTOE   — archived symbolic-composition probe over the same fabric
#
#  Every run is a real CLI decide call (manual-parity), deterministic.
#  Output: data/zeroshot_v32_results.json + a printed table.
#  Interpretation rules of docs/ZEROSHOT.md apply: confidence = concentration,
#  never correctness; probes bound, they never establish.
# ============================================================================
import json
import subprocess
import sys
from pathlib import Path

BUILD = "./build/syfox"

EMAIL_SPAM = [
    # (state, expected) — tickets-fabric vocabulary overlap is the zero-shot channel
    ("Congratulations you are our lucky winner click this link to claim your free prize now", "spam"),
    ("Urgent your account will be suspended verify your payment details immediately", "spam"),
    ("Win a free holiday click here offer expires today claim now", "spam"),
    ("Hi team the invoice for this month is attached please confirm the payment date", "ham"),
    ("Can we move the meeting to tomorrow morning and add the new project plan", "ham"),
    ("Thanks for your help with the refund the money arrived in my account today regards", "ham"),
]
SMS_SPAM = [
    ("FREE entry to win 1000 cash prize text WIN to the number now", "spam"),
    ("Your mobile prize is waiting claim your reward click the link urgent", "spam"),
    ("Hey are we still meeting at the pub tonight let me know", "ham"),
    ("Running late sorry be there in ten minutes see you soon", "ham"),
    ("Can you send me the photos from the weekend thanks", "ham"),
    ("Your account balance is low top up now to avoid charges", "spam"),
]
# snake: (state, safe set, expected single answer or the user's archived winner)
SNAKE = [
    ("snake game the food is directly ahead of the head move right", "right"),
    ("snake game a wall is directly ahead moving right hits the wall", "up/down"),
    ("snake game the body blocks the right side so right is lethal", "up/down/left"),
    ("snake game the food is behind while the head moves right", "up/down"),
    ("snake game the food is at the wall and right is lethal", "up/down"),
    ("snake game long term survival avoid the walls keep space", "up"),
    ("snake game only one safe move remains all others hit walls", "down"),
    ("snake game shortest safe route to the food", "up"),
]
TICTACTOE = [
    ("tic tac toe board positions 1 to 9 X holds 1 and 2 O holds 4 and 5 X to move "
     "three in a row wins the game the top row 1 2 3 completes with 3", "3"),
]
MOVES = ["up", "down", "left", "right"]

DIR_Q = json.dumps({"move": {"type": "choice", "instructions": "which move is safe",
                             "criteria": {
                                 "up": "move up away safe escape",
                                 "down": "move down away safe escape",
                                 "left": "move left away safe escape",
                                 "right": "move right toward food ahead"}}}, sort_keys=True)
TTT_Q = json.dumps({"cell": {"type": "choice", "instructions": "which cell wins now",
                             "criteria": {str(i): f"place at position {i}" for i in range(1, 10)}}},
                   sort_keys=True)


def spam_q() -> str:
    return json.dumps({"spam": {"type": "choice", "instructions": "is this message spam",
                                "criteria": {
                                    "spam": "free prize win click offer claim urgent account suspended verify now limited",
                                    "ham": "meeting schedule invoice project team regards thanks attached plan tomorrow"}},
                       }, sort_keys=True)


def run(model: str, state: str, q: str, extra=None):
    cmd = [BUILD, "decide", "--model", model, "--state", state, "--questions", q,
           "--energy-norm", "--retrieval-topk", "0"]
    if extra:
        cmd += extra
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        return {"error": out.stderr.strip()[:200]}
    return json.loads(out.stdout)


def tally(model: str, rows, q, expect_key):
    res = []
    for state, expected in rows:
        r = run(model, state, q)
        if "error" in r:
            res.append({"state": state[:40], "error": r["error"]})
            continue
        a = r["answers"]["spam"] if "spam" in r["answers"] else next(iter(r["answers"].values()))
        if "choice" in a:
            pick, conf = a["choice"], a["confidence"]
        else:
            pick, conf = (None, 0.0)
        res.append({"state": state[:50], "expected": expected, "picked": pick,
                    "conf": conf, "deferred": a.get("deferred", False)})
    ok = sum(1 for r in res if r.get("picked") and r["picked"] in str(r["expected"]).split("/"))
    return {"accuracy": round(ok / len(res), 3) if res else 0.0, "rows": res}


def main() -> None:
    models = {
        "email_tickets_plain": "model-tickets-big-latin",
        "email_tickets_sem": "model-tickets-big-latin-sem",
        "sms_tickets_plain": "model-tickets-big-latin",
        "sms_tickets_sem": "model-tickets-big-latin-sem",
        "snake_giant_plain": "model-giant-latin",
        "snake_giant_sem": "model-giant-latin-sem",
    }
    report = {}
    report["email_spam"] = tally(models["email_tickets_plain"], EMAIL_SPAM, spam_q(), "spam")
    report["email_spam_sem"] = tally(models["email_tickets_sem"], EMAIL_SPAM, spam_q(), "spam")
    report["sms_spam"] = tally(models["sms_tickets_plain"], SMS_SPAM, spam_q(), "spam")
    report["sms_spam_sem"] = tally(models["sms_tickets_sem"], SMS_SPAM, spam_q(), "spam")
    report["snake"] = tally(models["snake_giant_plain"], SNAKE, DIR_Q, "move")
    report["snake_sem"] = tally(models["snake_giant_sem"], SNAKE, DIR_Q, "move")
    report["tictactoe"] = tally(models["snake_giant_plain"], TICTACTOE, TTT_Q, "cell")
    report["tictactoe_sem"] = tally(models["snake_giant_sem"], TICTACTOE, TTT_Q, "cell")

    Path("data/zeroshot_v32_results.json").write_text(json.dumps(report, indent=1))
    for fam in ["email_spam", "email_spam_sem", "sms_spam", "sms_spam_sem",
                "snake", "snake_sem", "tictactoe", "tictactoe_sem"]:
        r = report[fam]
        print(f"{fam:22s} acc {r['accuracy']:5.2f}  " +
              "  ".join(f"{x['picked']}({x['conf']:.3f})" for x in r["rows"][:4]))
        for x in r["rows"]:
            if "error" in x:
                print("   ERROR:", x)


if __name__ == "__main__":
    main()
