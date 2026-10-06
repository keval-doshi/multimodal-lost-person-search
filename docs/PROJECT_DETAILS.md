# MultiModal Lost Person Search System Using Image and Text Query Across Multiple Cameras

**Keval Hirenbhai Doshi** — M01086516
MSc Data Science and Artificial Intelligence, Middlesex University
Module CST4275 · Supervisor: Dr Krishnadas Nanath · September 2026

---

## 1. What this project is

When someone goes missing, the search usually starts with two things: a photograph, and a
description of what they were wearing. Camera networks record continuously but cannot easily
be searched by either.

This project builds and evaluates a system that searches recorded footage from six cameras
using a photograph, a written description, or both. It detects people in whole uncropped
frames, encodes them and the query with frozen pretrained models, combines the two sources of
evidence, refines the ranking, and returns a sequence of sightings across cameras.

The research question is not "does it work" but **when is a description worth more than a
photograph** — and the answer turns out to depend on which image encoder you use and how
small the people are in frame.

---

## 2. The question

Three research questions drive the work:

- **RQ1** — Can people be detected and re-identified across several non-overlapping cameras
  when only whole, uncropped frames are available?
- **RQ2** — How should a written clothing description be combined with a photographic query,
  and which fusion strategy works best?
- **RQ3** — How does falling image resolution change the value of a description relative to a
  photograph?

RQ3 is the centre of the work. The original hypothesis was that a **quality-aware fusion
rule** — shifting weight toward the description as the image degrades — would beat a fixed
balance. That hypothesis was tested and mostly failed; see §10.

---

## 3. The answer, in short

**The value of a written description is conditional on the strength of the image encoder and
on the resolution of the footage.**

- With a **conventional re-identification model** (OSNet), a description becomes *significantly
  better* than a photograph once people are 32 pixels tall or smaller, and adding it to the
  photograph gains **+7.6 to +12.3 mAP points at every resolution tested**.
- With a **strong vision-language image encoder** (CLIP/IRRA), the photograph stays ahead at
  every resolution and the description adds **under half a point** — and only when the fusion
  weight is tuned for the configuration.
- For both encoders, the best balance shifts toward the description as resolution falls.
- The best system reaches **60.88 end-to-end mAP** on normal footage and **42.84** on a
  mixed-resolution gallery — but its gain comes from a **second photograph and re-ranking**,
  not from the description.

A practical consequence: if you have a strong image encoder and can obtain a second
photograph, do that before investing in collecting descriptions. If you are running a
conventional re-ID backbone, or your footage is genuinely low-resolution, a description is
worth a great deal.

---

## 4. Data

| Dataset | Role | Scale |
|---|---|---|
| **PRW** (Person Re-identification in the Wild) | Primary benchmark — whole uncropped frames from six near-synchronised campus cameras | 11,816 frames, 932 identities, 25,062 annotated pedestrians in the test split |
| **PRW-TBPS** | Written clothing descriptions attached to PRW pedestrians — **two independently written descriptions per test search** | matched to PRW boxes |
| **Market-1501** | Independent instrument check for the appearance model | 3,368 test queries |

The **two independent descriptions per search** are what make the description-sensitivity
analysis (§9.6) possible, and they are the reason PRW-TBPS was chosen over larger multimodal
resources. M²ReID (from PS-ReID) has full-scene image-and-text queries, but its descriptions
are GPT-4o-generated with one per query, and it removed pedestrians under 75 pixels — exactly
the regime this project studies.

All three datasets are public. **No component was trained**; every model is used with
published weights.

---

## 5. Pipeline

```
whole camera frames
        │
        ▼
  [1] DETECTION            YOLO11m, confidence 0.25
        │                  11,816 frames → 66,557 person boxes          (17 min)
        ▼
  [2] GALLERY FILTER       second threshold 0.5 before encoding
        │                  → 50,254 crops enter the searchable gallery
        ▼
  [3] ENCODING             OSNet x1.0  (MSMT17 weights)  → 512-d appearance descriptor
        │                  IRRA        (CLIP ViT-B/16, CUHK-PEDES)  → image + text embeddings
        │                                                                (16 min, both models)
        ▼
  [4] FUSION               z-score both similarity streams, then
        │                  score = α · photo + (1 − α) · description      ← contribution
        ▼
  [5] RE-RANKING           k-reciprocal (Zhong et al. 2017)               ← contribution
        │                  k1=20, k2=6, λ=0.3, applied to top 200
        │                  candidates below 200 keep their original order
        ▼
  [6] SIGHTING SEQUENCE    group by camera; collapse candidates within
                           25 frames in the same camera + segment
```

**Stages 4 and 5 are where the project makes its own contribution.** Stages 1–3 are cached and
run once; every experiment reads from disk, which is why a fusion change costs seconds rather
than half an hour.

### Simulating distance

Distance is simulated by degrading the **gallery only** (never the query) down a resolution
ladder. Each level degrades only crops already taller than the target, so the proportion
genuinely reduced is reported alongside every result:

| Level | Target height | Crops actually degraded |
|---|---|---|
| Native | unchanged | — |
| 1 | 96 px | 78.8% |
| 2 | 64 px | 91.1% |
| 3 | 48 px | 96.9% |
| 4 | 32 px | 99.7% |
| 5 | 24 px | 100% |
| 6 | 16 px | 100% |

A **mixed-resolution gallery** is also used. Each gallery crop is assigned independently and
uniformly at random to one of the seven levels above, with a fixed seed, so a single gallery
holds people at every simulated distance at once. It approximates a real camera network where
some people are near and some far, and it is the harder of the two: every result on it is lower
than the corresponding result on the native gallery.

### Runtime (single RTX 4060, 8 GB)

| Stage | Scale | Time |
|---|---|---|
| Detection, whole frame | 11,816 frames | 17 min |
| Detection, tiled | 1,175 frames | 5 min |
| Encoding | 50,254 crops, two models | 16 min |
| Degradation | 27,778 crops × 7 levels | 50 min |
| Retrieval experiments | per configuration | seconds to minutes |

---

## 6. Evaluation protocol — and the four corrections

**This is the most important methodological part of the project.** An audit of the evaluation
code partway through found four flaws, all of which inflated results. Every retrieval number
in the dissertation was regenerated after the corrections; earlier figures are superseded.

| # | Flaw in the original protocol | Correction |
|---|---|---|
| 1 | *Every* candidate from the query camera was removed | Remove only candidates that are **both the query person and from a query camera**; other people stay as distractors |
| 2 | Average precision counted only people the detector found; searches whose target was never detected were **skipped** | **End-to-end mAP** multiplies conditional mAP by the fraction of annotated appearances detected, so a missed person counts against the result and a never-detected target scores zero |
| 3 | Confidence intervals resampled **searches**, though one person contributes several | **Identity-level (cluster) bootstrap** — resample people, all their searches together |
| 4 | In re-ranking, every candidate below the top 200 was given the same score, so their order was arbitrary | Top 200 re-ordered; everything below **keeps its original fused order** |

The corrections **lowered several headline numbers and did not reverse the main findings** —
but they did change the story. Under the corrected protocol the originally proposed
OSNet-based system (52.68) is *significantly worse* than simply using the vision-language
encoder with two photographs (56.02). That is reported rather than hidden.

### Metrics

- **End-to-end mAP** is the primary metric. The purpose is to build a record of sightings
  across a network, so every appearance matters, and a person the detector missed is missed by
  the search.
- **Conditional mAP** (detected appearances only) is reported alongside for comparability.
- **Rank-1** is reported because it is the figure most commonly quoted in the literature.

### Statistics

Percentile bootstrap, **5,000 resamples, resampling identities not searches**. Comparisons are
paired — the same resampled people for both methods. Detection recall uses binomial (Wilson)
intervals instead.

Calibration was checked on synthetic data: the identity-level bootstrap covered the true mean
in 93.5% of 200 trials against a nominal 95%, while resampling searches individually produced
intervals **about half as wide** — which would have overstated the precision of every result.

### Splits

The dataset provides no validation split, so the 450 test identities are divided **by identity**
(seed 0): 225 for fitting, 225 for reporting. Splitting by identity rather than by image is
essential — splitting by image would put different photographs of the same person on both
sides.

Sixteen held-out searches have no annotated appearance outside their query cameras, so there
is nothing to find; they are excluded. Eight people had only such searches. **Reported figures
therefore come from 1,016 searches by 217 people.**

Every parameter is fitted on validation identities; every reported figure comes from held-out
identities.

---

## 7. Two system configurations

The complete system is reported in **two configurations**, and the distinction matters because
the description's contribution is small enough that the fusion weight decides whether it is
detectable:

| Configuration | Fusion balance α | Native | Mixed |
|---|---|---|---|
| **Shared balance** — one value for every method in the comparison, so all face identical settings | 0.80 | **60.88** | 42.84 |
| **Re-tuned balance** — fitted on validation identities for this exact configuration | 0.95 | 61.17 | 42.91 |

Headline figures quoted elsewhere are the **shared-balance** ones unless the re-tuned is named.

---

## 8. Results

### 8.1 Detection (RQ1)

Detection sets a ceiling no later stage can raise.

| Person height (px) | Annotated | Recall, all detections | Recall, search gallery |
|---|---|---|---|
| below 64 | 72 | 90.3 [81.3, 95.2] | 59.7 [48.2, 70.3] |
| 64–96 | 1,636 | 69.1 [66.8, 71.3] | **50.2 [47.8, 52.6]** |
| 96–128 | 5,662 | 84.4 [83.4, 85.3] | 74.5 [73.3, 75.6] |
| 128–192 | 7,227 | 93.4 [92.8, 93.9] | 87.0 [86.2, 87.8] |
| 192–256 | 3,788 | 98.5 [98.1, 98.8] | 96.4 [95.8, 96.9] |
| above 256 | 6,677 | 99.2 [99.0, 99.4] | 98.6 [98.3, 98.9] |
| **all** | **25,062** | **92.1 [91.8, 92.4]** | **86.2 [85.8, 86.6]** |

Only **half** the pedestrians between 64 and 96 pixels ever enter the searched gallery. That
loss happens before any matching, and no re-identification method can recover it.

### 8.2 Single-source retrieval, native gallery

| Query | End-to-end mAP | Conditional mAP | Rank-1 |
|---|---|---|---|
| Photograph, appearance model | 29.04 [26.19, 31.96] | 30.83 | 52.95 |
| Photograph, vision-language model | 45.71 [42.37, 49.03] | 48.70 | 65.06 |
| Description | 26.11 [23.18, 29.00] | 27.87 | 37.60 |
| Two photographs, appearance model | 38.29 [34.98, 41.64] | 40.79 | 66.73 |
| Two photographs, vision-language model | 56.02 [52.53, 59.46] | 59.82 | 78.15 |
| Two descriptions, averaged | 33.51 [30.32, 36.70] | 35.79 | 48.03 |

### 8.3 Resolution — the core result (RQ3)

End-to-end mAP against gallery resolution, one photograph vs one description:

| Level | Appearance | Vision-language | Description | Appearance − description | Vision-language − description |
|---|---|---|---|---|---|
| native | 30.45 | 46.18 | 27.81 | +2.64 [+0.19, +4.99] | +18.36 [+16.35, +20.40] |
| 96 px | 30.01 | 47.49 | 28.26 | +1.74 [−0.63, +4.10] | +19.22 [+17.21, +21.29] |
| 64 px | 28.17 | 46.74 | 27.20 | +0.97 [−1.40, +3.28] | +19.54 [+17.56, +21.60] |
| 48 px | 25.03 | 44.30 | 25.20 | −0.17 [−2.47, +2.10] | +19.09 [+17.14, +21.09] |
| **32 px** | 17.48 | 35.94 | 19.57 | **−2.09 [−4.04, −0.14]** | +16.38 [+14.45, +18.36] |
| 24 px | 11.23 | 26.11 | 14.51 | −3.28 [−4.91, −1.67] | +11.60 [+9.96, +13.36] |
| 16 px | 3.60 | 13.84 | 8.74 | −5.13 [−6.50, −3.91] | +5.10 [+3.86, +6.47] |

**The crossover is at 32 pixels** — for the appearance model. The vision-language encoder never
crosses; it stays ahead of the description at every level.

Retention at 16 px relative to native: appearance **11.8%**, vision-language **30.0%**,
description **31.4%**. The description degrades most gracefully because it does not degrade at
all — only the gallery does.

### 8.4 Fusion balance and description gain

Left: appearance model. Right: vision-language model. `*` = interval excludes zero.

| Gallery | α | Photos only | With description | Gain | α | Photos only | With description | Gain |
|---|---|---|---|---|---|---|---|---|
| native | 0.50 | 38.29 | 48.81 | **+10.52** * | 0.95 | 56.02 | 56.32 | +0.30 * |
| mixed | 0.50 | 25.36 | 33.99 | +8.63 * | 0.90 | 39.30 | 39.59 | +0.29 * |
| 48 px | 0.50 | 30.90 | 43.15 | **+12.25** * | 0.90 | 51.33 | 51.67 | +0.34 * |
| 32 px | 0.40 | 21.46 | 33.38 | +11.92 * | 0.90 | 41.24 | 41.66 | +0.42 * |
| 24 px | 0.40 | 13.95 | 23.94 | +9.99 * | 0.85 | 29.67 | 30.16 | +0.50 * |
| 16 px | 0.25 | 3.73 | 11.29 | +7.56 * | 0.70 | 15.24 | 15.73 | +0.49 * |

The tuned α **shifts toward the description as resolution falls** for both encoders (0.50 → 0.25,
and 0.95 → 0.70). The mechanism the original hypothesis predicted is real — it just isn't worth
much for the strong encoder.

### 8.5 Complete systems and all baselines

| Method | Native gallery | Mixed-resolution gallery |
|---|---|---|
| Photograph, appearance | 29.04 [26.19, 31.96] | 19.16 [17.11, 21.27] |
| Photograph, vision-language | 45.71 [42.37, 49.03] | 32.09 [29.45, 34.67] |
| Description | 26.11 [23.18, 29.00] | 18.94 [16.72, 21.20] |
| Two photographs, appearance | 38.29 [34.98, 41.64] | 25.36 [22.90, 27.88] |
| Two photographs, vision-language | 56.02 [52.53, 59.46] | 39.30 [36.40, 42.18] |
| System, appearance model | 52.68 [48.95, 56.41] | 37.23 [34.15, 40.35] |
| **System, vision-language model** | **60.88 [57.20, 64.43]** | **42.84 [39.68, 45.99]** |

The appearance-model system is **significantly worse** than two vision-language photographs
alone: −3.35 [−5.06, −1.70] natively. This contradicts the project's original design, in which
OSNet was the centrepiece, and is reported as such.

### 8.6 Where the gain comes from — component analysis

Each component removed from the complete system in turn (removal, not addition, so overlapping
gains don't double-count):

| Change to the system | Native gallery | Mixed-resolution gallery |
|---|---|---|
| Remove fusion | −9.25 [−10.96, −7.58] * | −8.72 [−9.98, −7.46] * |
| Remove second photograph | −6.00 [−6.88, −5.13] * | −4.11 [−4.75, −3.48] * |
| Remove second description | −3.75 [−4.57, −2.96] * | −2.86 [−3.48, −2.24] * |
| Remove re-ranking | −3.96 [−4.53, −3.39] * | −3.26 [−3.79, −2.75] * |
| Add query expansion | −0.46 [−0.89, −0.06] * | −0.50 [−0.82, −0.19] * |
| Add resolution matching | +0.06 [−0.10, +0.22] | +0.91 [+0.48, +1.34] * |

**Description's contribution to the vision-language system**, against one baseline — two
photographs re-ranked without any description, **60.86**:

| Configuration | Result | Gain |
|---|---|---|
| Shared balance 0.80 + re-ranking | 60.88 | +0.02 [−0.57, +0.58] — not significant |
| Re-tuned balance 0.95 + re-ranking | 61.17 | +0.31 [+0.11, +0.55] — significant |
| The two configurations compared directly | — | +0.29 [−0.16, +0.76] — **not significant** |

That last row matters: one gain being significant and the other not is **not** the same as a
significant difference between them. The results are *consistent with* sensitivity to the
fusion weight; they do not establish it.

### 8.7 Description sensitivity (exploratory)

PRW-TBPS gives two independently written descriptions per search, which allows a question
almost never asked: **how much does the answer depend on who wrote the description?**

| Measure | Result |
|---|---|
| Mean difference in mAP between a search's two descriptions | **16.73 points** |
| The same, once combined with a photograph | 10.17 points |
| Reduction from combining with a photograph | +6.56 [+5.11, +8.05] |
| Searches whose two descriptions differ by more than 40 points | 11.9% (121 of 1,016) |
| Description-only mAP: first description → both averaged | 26.11 → 33.51 |
| Gain from averaging the two descriptions | **+7.40 [+6.20, +8.61]** |

Two people describing the same person produce results differing by 16.7 points on average.
**Collecting a second description is worth more than most algorithmic choices in this project.**

### 8.8 Confidence signal (exploratory)

The top fused score separates reliable from unreliable searches:

| Confidence band | Fused retrieval | Complete system |
|---|---|---|
| lowest fifth | 20% [14, 27] | 29% [21, 37] |
| second | 46% [39, 54] | 63% [56, 71] |
| middle | 64% [57, 71] | 82% [76, 88] |
| fourth | 77% [71, 83] | 81% [74, 87] |
| highest fifth | 83% [76, 90] | 86% [79, 92] |
| **overall** | 59.9% | 70.4% |

**Important caveat, stated throughout the dissertation:** the signal itself was chosen in an
earlier exploratory analysis that ran over *all* test searches, so the held-out searches took
part in selecting it. Only the quintile thresholds were fitted independently. This is an
**exploratory result with a partially independent evaluation** and would need confirming on a
fresh set of identities.

---

## 9. Failure analysis

Where the strongest system goes wrong, on the native gallery (exploratory):

| Group of searches | Searches | Rank-1 | End-to-end mAP | Conditional mAP |
|---|---|---|---|---|
| 75% or fewer of appearances detected | 74 | 78.4% | 43.1 | 64.1 |
| More than 75%, under 99% detected | 463 | 80.8% | 60.0 | 65.7 |
| 99% or more detected | 479 | 73.7% | 64.5 | 64.5 |
| 1 to 5 appearances to find | 63 | 46.0% | 46.3 | — |
| 6 to 10 appearances to find | 145 | 58.6% | 50.0 | — |
| 11 to 20 appearances to find | 316 | 76.6% | 60.4 | — |
| More than 20 appearances to find | 492 | 87.2% | 66.2 | — |

Two readings. First, where detection misses people, the loss is carried **directly** into the
search result — conditional mAP barely moves while end-to-end mAP collapses. Second, **people
who appear rarely are much harder to find** (46% Rank-1 with 1–5 appearances vs 87% with more
than 20), which is precisely the case that matters most in a real missing-person search.

The first result is wrong in roughly **a quarter** of searches even for the strongest system.
The output is a shortlist for human review, not an identification.

---

## 10. What did not work

Negative results are reported rather than quietly dropped:

| Component | Result | What it establishes |
|---|---|---|
| **Slicing-aided inference (SAHI)** | +0.9 points on small people, losses on larger ones | A technique's benefit depends on the data containing the condition it targets; PRW has few very small pedestrians |
| **Per-candidate quality weighting** — the original proposed contribution | No gain (appearance model); +0.34 for the vision-language model but **in the opposite direction to the hypothesis**, placing *more* weight on the photograph for poorer crops | This rule family did not realise the proposed idea; the mechanism remains open |
| **Query expansion** | −0.46 and −0.50 points | Adding a component that uses the same neighbourhood evidence as re-ranking did not help |
| **Description with the vision-language encoder** | +0.31 with the balance tuned; +0.02 at the shared balance | A strong image encoder and a second photograph leave little for a description to add in late fusion |

The quality-aware rule is **partly rescued**: chosen *per resolution condition* rather than per
candidate, it gives +0.48 [+0.28, +0.69] for the appearance model and +0.17 [+0.02, +0.31] for
the vision-language model — small but significant. At 16 px the per-condition gain rises to
+1.86 [+1.24, +2.54].

Fusion controls were run to show the evaluation *can* detect a gain where one exists, which
makes it less likely that a null result is an artefact of the implementation.

---

## 11. Limitations

- Degradation is **synthetic**. Real distant footage carries motion blur, compression artefacts
  and sensor noise that downsampling does not reproduce.
- All retrieval results come from **one benchmark on one campus**. Checked on two encoders, not
  two camera networks.
- The vision-language model has **probably seen imagery from the same campus** during training
  (CUHK-PEDES draws on Market-1501, recorded on the same campus as PRW). Whether this affects
  the *shape* of its resolution curve, not just its level, was not tested.
- The protocol keeps unlabelled people and false detections as distractors, so numbers are
  **not directly comparable with official PRW results**.
- Re-ranking is applied to the top 200 — a standard approximation; the full method may behave
  differently.
- The **confidence signal's selection was not independent** of the held-out searches (§8.8).
- Description sensitivity mixes differences in *content* with differences in *wording*; the
  length relationship is correlational and rests on 16 searches in its longest group.
- The released data provide **no usable timestamps**, so the output is an ordered sequence of
  sightings rather than clock times, and temporal grouping can merge different people seen close
  together.
- The prototype interface implements the earlier single-photograph configuration, not the
  complete system.
- The **literature review is structured, not systematic** — search strings, dates and screening
  counts were not logged, so its selection cannot be reproduced from Chapter 2. The retained set
  of 41 studies in Appendix B is the record it does provide.
- The apparent sensitivity of the description's contribution to the fusion weight is **suggested
  by the results, not established** (§8.6).

---

## 12. Ethics

The system has **no dedicated face-recognition component** — no face detector, face descriptor
or face database. But the encoders receive whole-person crops that **include the head**, so
facial appearance may contribute to the features they produce, and whole-body appearance
features used to single out a person can themselves be personal data under data-protection law.
No claim is made about what the networks do or do not encode; they were not examined.

At these camera distances a face spans roughly 23 pixels with about 7 between the eyes — well
below what reliable facial identification needs. Matching relies mainly on clothing, build and
general appearance. This **reduces rather than removes** the privacy risk.

A deployment would need a lawful basis and specific authorisation for each search, human review
of every result, limited retention, an audit trail, and disclosure of the confidence attached
to each result.

---

## 13. Repository contents

See the repository structure in the main `README.md`. The dissertation itself is not included
in this repository.

### Verification

`verify_tables.py` reads every number out of the dissertation's tables and compares it against
the retained output files. Bootstrap intervals are recomputed with `cboot()` **exactly as
defined in `code/step1_corrected_eval.py`** — numpy `default_rng`, seed 0, 5,000 identity-level
resamples, percentiles at 2.5/97.5 — so every interval is reproducible with the submitted code
at its default settings.

```bash
pip install python-docx pandas numpy
python verify_tables.py path\to\Dissertation.docx
```

**Covered:** Tables 5.1–5.7, 5.9, 5.10, 6.1, and the description-contribution comparisons of
Sections 5.9.1 and 6.4 — including a check that both re-ranked comparisons rest on the same
no-description baseline (60.86).

**Table 5.8 and Figure 5.6** are covered when the confidence outputs are present in `results/` — see §15.

---

## 14. Reproducing from scratch

`config.json` records every setting: fusion balance 0.45 (appearance) and 0.80
(vision-language) for the system comparison, k-reciprocal k1=20 k2=6 λ=0.3 over the top 200,
query expansion over the top 8, 5,000 identity-level resamples, seed 0.

With PRW, PRW-TBPS and Market-1501 in `dataset/`, build the caches:

```bash
python verify.py                 --root .\dataset
python phase2_detect.py          --root .\dataset --mode full
python phase2_eval.py            --root .\dataset --split test
python phase3_sanity.py          --root .\dataset
python phase3_encode.py          --root .\dataset
python phase4_resolution.py      --root .\dataset
python phase7_resmatch.py        --root .\dataset --encode
```

Then regenerate every Chapter 5 retrieval result from those caches:

```bash
python step1_corrected_eval.py   --root .\dataset
python step1b_alpha.py           --root .\dataset
python step1c_condition_alpha.py --root .\dataset
python step1d_remaining.py       --root .\dataset
python step1e_confidence.py      --root .\dataset
```

Detection and encoding are the expensive stages and run once. Everything after reads from
`cache/`; all evaluation outputs go to `cache/step1/`.

**Superseded scripts:** `phase4_map_ci.py`, `phase5_ci.py`, `phase7_algorithms.py` and
`phase8_full.py` implement the *pre-correction* protocol. Their outputs are kept for the record
and are **not** used for any reported result.

---

## 15. Confidence-signal outputs

`e_confidence.txt` and `e_confidence_perquery.csv` are written by `step1e_confidence.py`
(behind **Table 5.8 and Figure 5.6**). They were not kept in the original submission archive and
were later regenerated by re-running the script against the caches; the regenerated bands match
Table 5.8 exactly (fused retrieval 20 / 46 / 64 / 77 / 83%, overall 59.9%; complete system
29 / 63 / 82 / 81 / 86%, overall 70.4%). When both files are in `results/`, `verify_tables.py`
checks Table 5.8 as well.

---

## 16. Development and review history

The project went through an unusual amount of self-correction, which is worth recording because
it shapes what the results mean.

1. **Original evaluation** produced a clean-looking story built on the OSNet system.
2. **Protocol audit** found four flaws (§6). All were corrected and every retrieval result
   regenerated. Several headline numbers fell, and the recommended system turned out
   *significantly worse* than a simpler baseline.
3. **Confidence signal retest** — an earlier analysis had tested the *gap between the top two
   scores* rather than the pre-specified *top score*. A single confirmatory rerun was written to
   test the right signal; this was the only retest, to avoid signal-fishing.
4. **Three external review rounds** then caught, in order:
   - Overclaimed confidence validation — corrected to state that the signal's *selection* was
     not independent even though its thresholds were.
   - A **mislabelled comparison**: +0.34 was the description's gain *before* re-ranking against
     a 56.02 baseline, not the gain inside the re-ranked system. The arithmetic gave away a
     baseline of 60.54 where it should have been 60.86. Corrected, and the missing number
     (+0.02) computed.
   - A **bootstrap interval computed with the wrong function** — an ad-hoc resampler rather than
     the project's own `cboot()`. Recomputed; every interval now uses the submitted code.
   - An **overstated mechanism** ("over-weighting cancels out what it contributes") — replaced
     with the direct test, which found no significant difference between the two configurations.
   - A README that **overstated verification coverage** — Tables 5.7 and 6.1 were then actually
     verified, raising coverage from 173 to 223 checks (226 after the final additions), and the remaining gap stated plainly.

The pattern worth noting: **each round of scrutiny made the claims smaller and the evidence
stronger.** The protocol corrections cost the project its headline result and gained it a more
honest one.

---

## 17. Future work

The clear next step is a **jointly trained multimodal model** rather than late fusion over
frozen encoders — specifically one that learns the modality weighting *internally*, conditioned
on crop resolution, and is trained with resolution augmentation so it sees the whole ladder.
That would directly test whether learned resolution-conditioned fusion beats the tuned-α
baseline measured here.

The real bottleneck is **training data**, not compute: PRW-TBPS is an evaluation resource, not a
training set, and the available full-scene multimodal datasets either lack small pedestrians or
use machine-generated captions. On 8 GB, LoRA/adapters plus bf16, gradient checkpointing and
**GradCache** (for large effective contrastive batches) make it feasible; renting a cloud GPU
would be faster and cheap.

Other directions: real distance and a second camera network; explaining *why* the encoders
differ so much; labelled failure cases with deliberately inaccurate descriptions; and scaling to
longer footage with real timestamps.

The evaluation harness built here — corrected protocol, end-to-end mAP, identity-level
bootstrap, the resolution ladder, 226 automated checks — is the asset that makes any of this
tractable. A new encoder drops in where `phase3_encode.py` sits and everything downstream still
works.
