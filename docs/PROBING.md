# Probing a SyFox model — methodology (v3.1)

Handcrafted probes are diagnostic instruments, not benchmarks. Used wrong,
they produce misleading conclusions in BOTH directions. This doc encodes what
the v3.1 probe experiments measured (real UCI SMS spam, one fabric, one hidden
set, three decide-side criteria wordings):

| decide-side criteria | hidden top-1 | mean margin (p1-p2) |
|---|---|---|
| matched (the schema the fabric was taught) | 0.160 | 0.444 |
| reworded (handcrafted, abstract) | **0.842** | 0.043 |
| empty (labels only) | 0.422 | 0.017 |

The SAME fabric, the SAME rows. The criteria text at decide is a first-class
input to the readout — it selects WHICH nodes the question probes, and
therefore which signal (or noise) the field returns.

## Rules for handcrafted probes

1. **Never conclude from margins under reworded criteria alone.** Reworded,
   abstract criteria suppress direct-term harvest and often raise accuracy —
   but they also shrink margins into the tie zone (0.01-0.05 with confidence
   ~0). A tie is a real answer: it means "the field cannot separate these
   options through THIS probe wording".

2. **Use `--defer-margin P` when ties must not masquerade as decisions.**
   The substrate always decides (physics untouched); the flag discloses
   margin below P as `deferred: true, reason: "low_margin"` instead of a
   confident-looking label on 50.9/49.1 splits. Same knob on the server:
   `"options": {"defer_margin": 0.1}`.

3. **Sweep the criteria wording — one fabric, three conditions:**
   matched / reworded / empty. Agreement between conditions is evidence about
   the fabric; disagreement is evidence about the wording. A fabric that only
   works under its training schema is overfit to its own question format.

4. **Match the training schema when reproducing benchmark numbers.** The
   held-out numbers a model reports (e.g. SMS 0.937) were measured under the
   schema the rows carry. Handcrafted probes with new wording measure a
   different system — not the one the number describes.

5. **Handcrafted probes bound, never establish.** They can demonstrate a
   failure (a false negative on an obvious scam) and motivate a held-out run;
   they cannot estimate accuracy. Formal conclusions come from
   `bench --eval <held-out>` on the real dataset.

## Why ties happen (mechanism, measured)

At learn time the substrate wires `state words -> label + training-criteria
words` (Hebbian). At decide time the probe is `label + criteria-as-given`,
read out of the settled state field as: direct energy at probe-token nodes +
a weighted-mean one-hop over their lanes. Two failure modes dominate:

- **Direct-term harvest**: probe words that appear in the state add energy
  regardless of class meaning. Class-pure bigrams containing globally common
  words ("cash abroad" -> "cash") let one option harvest from every state.
- **Anchor one-hop typicality**: the label token's lanes point at its class's
  typical vocabulary. This is the true carrier — and it shows through best
  when the probe criteria are abstract enough to avoid the harvest.

Opaque per-class anchors (`c00..c76`) + empty instructions at learn reduce
hub pollution; see the v3.1.0 banking77 ladder (0.029 -> 0.091 cal) for the
measured effect sizes.

## Reconciliation note for model-xl probes

The model-xl diagnostic (Bangla/English/SMS ties at ~0.50/0.50 with
`deferred: false`) is consistent with these mechanisms: the probes used new
criteria wording on a 14-domain fabric where every option's probe words carry
comparable mass. The fabric's own held-out numbers (SMS 0.937) were measured
under its trained schema. Next step on that side: re-run the probes with the
trained schema, sweep wording, and score the real held-out sets with
`--defer-margin` so ties are visible instead of silent.
