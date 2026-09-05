# Kyivan — Paper Outline (thesis flow, English)

Introduction is already drafted in the `.tex` — this file starts from
Related Work. Each numbered point is one thesis; points within a section
are ordered so each one sets up the next. Citation format: **(Source, page/
location)**.

---

## Related Work

Structured around the two approaches the field actually splits into for
this problem — **fine-tune an existing multilingual encoder** vs. **train
a character-level architecture from scratch** — since that's the choice
Kyivan itself had to make (last point below) and the two threads should
converge on that decision, not sit as an unordered list.

### MLM with fine-tuned BERT

1. The MLM pretraining objective — predict a token from bidirectional
   context — is structurally identical to lacuna restoration; this
   observation is what motivates using BERT-family models for restoration
   at all, fine-tuned or not. **(Devlin et al. 2018, "BERT", pp. 1–2, 4 —
   already `devlin2018`, cited in Introduction)**
2. **Already drafted in the `.tex`, verified against the paper, ready as
   is** (two trivial typos to fix on a pass: "assined" → "assigned",
   "training from scratch" → "trained from scratch"): fine-tuning a
   general multilingual encoder has worked for a comparably low-resource
   ancient language — mBERT restores Akkadian lacunae and *outperforms* a
   model trained from scratch on Akkadian alone; the Akkadian mBERT was
   never trained on cuneiform, only on the Latin transliteration, so it
   already reads Latin text fluently from its own pretraining; the authors
   also assigned mBERT's 99 available free tokens, closing the residual
   vocabulary gap, via WordPiece's max-likelihood algorithm. **(Lazar et
   al. 2021 — Latin-transliteration point: p. 3, §2.1, "Oracc contains
   Latinized transliterations of the cuneiform texts"; 99-token point:
   p. 4, §4.2, "we assign its 99 available free tokens, optimizing for
   maximum likelihood by the WordPiece tokenization algorithm")**
   Deliberately left as a plain report of what Lazar et al. did — **the
   "why this doesn't transfer to Kyivan" counterargument belongs in Data,
   not here** (Related Work reports on other systems; the paper's own
   design rationale, and the Zaliznyak dialect-diversity citation it
   leans on, is a Data-section thesis — see Data point 6 below).

### Architecture from scratch

3. The first deep-learning approach to ancient-text restoration was a
   from-scratch character/word BiLSTM seq2seq model trained on Greek
   epigraphy, substantially outperforming expert epigraphists (CER 30.1%
   vs. 57.3%) — the original evidence that from-scratch beats not just
   humans but the fine-tuning alternative in this domain. **(Assael et al.
   2019, "Pythia", pp. 1, 3–4)**
4. Its successor extended single-task restoration into three jointly
   trained tasks — restoration, geographic attribution, chronological
   attribution — via a sparse-attention Transformer, and introduced the
   geometric-distribution span-masking scheme Kyivan's own training regime
   reuses at the same parameter value. **(Assael et al. 2022, "Ithaca",
   pp. 2–3 architecture/results; pp. 6, 9 masking scheme)**
5. The next iteration replaced that torso with a purely character-level
   T5+RoPE model and was the first to handle gaps of *unknown* length, via
   a dedicated head plus non-sequential beam search over the whole gap —
   the direct architectural ancestor of Kyivan, including its tied
   input/output embedding projection. **(Assael et al. 2025, "Aeneas",
   pp. 2, 4–5, 8 results/corpus; p. 10 provinces; pp. 11, 14–15 RoPE/tied
   projection)**
6. Kyivan adopts RoPE from its original formulation — rotating query/key
   vectors to encode relative position instead of learned absolute
   embeddings — following that precedent. **(Su et al., "RoFormer",
   pp. 1–3)**
7. Kyivan is not a first attempt at this problem: the system and corpus
   this paper reports results on originate in earlier work by one of the
   authors, publicly released as the **LacunaBERT-slav** checkpoint (HSE
   affiliation per contact info: Maxim Eremeev, maeremeev@edu.hse.ru).
   **(Eremeev, diploma/thesis — placeholder, still need full citation:
   title, year, institution; model card: huggingface.co/BeRestoral/
   LacunaBERT-slav)** That citation will be replaced by his own
   forthcoming, birchbark-focused paper once published. Worth a
   data-availability sentence in the paper itself pointing at the released
   checkpoint/model card directly. This is the thread's natural endpoint —
   the "from scratch" side wins the choice framed in the first
   subsection, and this paper is the next step in that same lineage.

**Table needed here (end of subsection, closes the lineage visually):**
task-coverage comparison, one row per system, one column per capability —
Pythia / Ithaca / Aeneas / Kyivan (rows) × restoration, geographic/regional
attribution, chronological attribution, unknown-length gaps, non-sequential
beam search (columns), ✓/✗ cells. Makes points 3–7 skimmable at a glance
instead of only readable as prose. All the ✓/✗ values are already stated in
points 3–7 above — this is a formatting pass, not new research.

---

## Data

Per the colleague's skeleton, this section opens with a **table of
contents** — literally the source table below doubles as that: it's both
the section's overview and point 1's evidence.

1. The corpus pools 9 named source collections (a 10th, Byliny, carries no
   usable date/dialect metadata and is excluded from training):

   | Source | Dialect | Documents | Word tokens |
   |---|---|---|---|
   | Birchbark manuscripts | Novgorodian (NW) | 1,241 | 29,042 |
   | Epigraphy | Church Slavonic (CS) | 1,016 | 8,172 |
   | NKRYA (historical) | Old Russian / Novgorodian | 2,998 | 240,310 |
   | UD Old East Slavic – RNC | Old East Slavic (OES) | 322 | 143,369 |
   | UD Old East Slavic – Ruthenian | South-Western (SW) | 420 | 113,956 |
   | TOROT | Old Russian / Church Slavonic | ≥39 | ≥306,971 |
   | Pushkin House | Old Russian | ≥57 | ≥401,505 |
   | Sofia Chronicle | Church Slavonic (CS) | 153 | 1,049,794 |
   | Ostrog Bible | Church Slavonic (CS) | 76 | 800,605 |

   Total: **≥6,365 documents, ≥3.14M word tokens.** **(LacunaBERT-slav
   model card, huggingface.co/BeRestoral/LacunaBERT-slav, "Training data")**
   The card notes the corpus was hand-filtered after this table was
   compiled, with that filtering step "to be added" to the documentation —
   flag as an open item to resolve with Eremeev before citing exact counts
   in a submitted draft.
2. Three special tokens carry the model's notion of "missing text," and
   the distinction is load-bearing for interpreting Results: `[-]` — one
   masked character, always a loss target (synthetic or a real editorial
   reconstruction); `[#]` — a compressed gap of unknown length (synthetic
   at train time; a real, historically attested gap at its real position
   and length in Test B, taken from editorial brackets in the source
   transcription); `[UNK]`-equivalent unresolved lacunae are never
   synthetically masked or scored. Only letters and spaces are ever masked
   or predicted — punctuation is always left visible, so the model cannot
   restore missing punctuation. **(LacunaBERT-slav model card, "Masking";
   repo: `collator_v2.py`, `prepare_splits.py::process_test_b_line`)**

   **Table needed here (right after this point, glossary made concrete):**
   one row per token, columns Token / Meaning / Origin (synthetic vs. real)
   / Example. A worked example already exists in this file's Figures notes
   (`paper_prep_notes.md`, birchbark_1002) — reuse it: masked
   `н е [-][-]л а т [-] л [-]` → target `н е п л а т и л е`. Small table,
   but does more to make the three-token distinction stick than another
   paragraph would.
3. Two test splits probe two different questions: Test A applies the same
   synthetic masking used in training to intact text (general restoration
   capacity); Test B conceals real editorial reconstructions already
   present in the source markup (practical utility on genuinely damaged
   text) — Kyivan's own evaluation design, not inherited from Ithaca/
   Aeneas. The published model card reports Test B only; Test A numbers
   below are computed separately from the same run's raw prediction dump.
4. Region attribution targets four macro-dialect classes (OES/CS/NW/SW)
   rather than Aeneas's 62 administrative provinces, because Old
   Novgorodian scholarship is organized around dialect zones, not modern
   administrative regions. **(Zaliznyak 2004, pp. 7–17)** Date attribution
   uses 20 bins of 50 years (800–1799 AD), matching the field's own
   achievable dating precision once stratigraphy and palaeography are
   combined. **(Zaliznyak 2004, p. 21)**
5. The corpus is heavily imbalanced by source: birchbark letters — this
   paper's actual target domain — are short (median 130 characters,
   **Zaliznyak 2004, p. 22**) and are the smallest source by word-token
   count (29,042, under 1% of the ≥3.14M total) despite being the
   second-largest by document count, while Sofia Chronicle and Ostrog
   Bible alone contribute ≈1.85M word tokens between them (≈59% of the
   total) from under 230 documents combined. This is a live, uncorrected
   limitation — every aggregate restoration number in Results should be
   read against it.
6. **The corpus-design answer to the fine-tuning counterargument from
   Related Work.** Text is normalized (paleographic/orthographic cleanup —
   `normalize_historical_text` in the repo) before training, but
   normalization only reduces spelling noise within a document; it doesn't
   collapse the corpus onto one target orthography. Unlike Oracc's
   Akkadian, which is transliterated into a single uniform Latin scheme
   Lazar et al. could then close a residual vocabulary gap for with 99
   reserved tokens (Related Work, MLM point 2), this corpus spans several
   genuinely distinct dialects — Old East Slavic, Church Slavonic,
   Novgorodian, South-Western — that differ grammatically and
   orthographically, not just in surface spelling: "not fewer than five
   Slavic idioms" in the Novgorod land alone. **(Zaliznyak 2004, pp. 7–8,
   already cited in the Introduction — this is the same citation, reused
   for the corpus-heterogeneity argument specifically)** There's no single
   vocabulary gap of the Akkadian kind here to close with a fixed token
   budget; a from-scratch, character-level model sidesteps needing one.

---

## Methods

Per the colleague's skeleton, this section is now three subsections
(Date prediction / Dialect attribution / Text restoration), not a flat
list. Open with a short unlabeled lead-in covering what's shared across
all three heads, then let each subsection cover only what's specific to
it — don't repeat the shared-backbone description three times.

**Shared backbone (lead-in, before the subsections):** a BERT encoder
with every self-attention layer replaced by a RoPE-injected variant,
following RoFormer's formulation as adopted by Aeneas for this exact task
family — architecturally simpler than Aeneas itself, since Kyivan uses a
bidirectional encoder torso directly rather than a T5-based decoder.
**(Su et al.; Assael et al. 2025, pp. 11, 14–15)** Text is tokenized
purely at the character level, with no word-level representation —
necessary because restoration predicts individual characters, not
subwords, matching Aeneas's design and departing from Ithaca/Pythia's
word+char dual representation. **(Assael et al. 2025)** All heads below
train jointly on this one shared encoder, extending Ithaca/Aeneas's
joint multi-task design. **(Assael et al. 2022; 2025)**

**Figure needed here (pipeline diagram, Aeneas Fig. 2 style):** input text
tiles → normalization → BERT+RoPE torso ×16 → branches into the four
heads below, each rendered as its own output panel (restoration = char
tiles; date = 20-bin histogram, 800–1799 AD axis; region = 4-bar chart,
OES/CS/NW/SW; unk = expand/stop decision, visually flagged as the one head
with no Aeneas counterpart). Full build spec (what to keep/drop from
Aeneas's own figure, exact example text, Figma component notes) already
exists in `paper_prep_notes.md`'s Figures section — port it here rather
than re-deriving it, the earlier write-up is still current.

**Table needed here (architecture-size comparison, Kyivan vs. Aeneas):**

| | Aeneas (Latin config) | Kyivan |
|---|---|---|
| Parameters | ~26–28M (formula estimate) | 512 hidden — get exact count from Eremeev or compute from the released checkpoint |
| hidden / emb dim | 384 | 512 |
| FFN size | 1536 | 2048 |
| layers | 16 | 16 |
| attention heads | 8 | 8 |
| position encoding | T5 relative bias | RoPE |
| vocab (chars) | 32 (Latin) | 118 |
| max context | 768 | 1024 |

(Reused from an earlier prep pass — `paper_prep_notes.md` had a param
count of 51,564,175 for a *different*, since-superseded Kyivan config;
don't carry that number over, it's stale. Get the real parameter count for
the `kyivan_run5`/LacunaBERT-slav checkpoint specifically before this
table goes in a draft.)

### Date prediction

1. Pooled from the `[SOS]` position, a 20-bin distribution over 50-year
   buckets (800–1799 AD) — a **period, not one particular year**: matches
   the field's own achievable dating precision once stratigraphy and
   palaeography are combined for a birchbark document. **(Zaliznyak 2004,
   p. 21; LacunaBERT-slav model card, "Architecture")**
2. Loss weight 0.5 — down from a literal copy of Aeneas's Latin
   configuration, retuned after the date head's unbounded KL loss was
   suspected of dominating the shared encoder's gradient.

### Dialect attribution

3. Pooled from `[SOS]`, four macro-dialect classes (OES/CS/NW/SW) rather
   than Aeneas's 62 administrative provinces, because Old Novgorodian
   scholarship is organized around dialect zones, not modern
   administrative regions. **(Zaliznyak 2004, pp. 7–17)**
4. Loss weight 0.5 — same retuning rationale as date (2 above): both
   auxiliary heads were converging faster than restoration at Aeneas's
   original weights, so their share of the gradient was reduced.

### Text restoration

5. Two heads jointly handle restoration: a per-token **restore** head
   (tied to the character embedding matrix, following Aeneas's own
   tied-projection design — **Assael et al. 2025, pp. 14–15**) predicting
   the character at each `[-]` mask, and a per-token **unk** head (binary)
   deciding whether an unknown-length lacuna (`[#]`) should expand by one
   more character or stop.
6. Masking mirrors Ithaca's geometric-distribution span scheme at the same
   parameter value (p=0.1) **(Assael et al. 2022, pp. 6, 9)**, plus a
   Kyivan-specific edge-tear augmentation — masking a short span at a
   sequence's start or end — motivated directly by birchbark's actual
   damage pattern of deliberate tearing by an addressee. **(Zaliznyak
   2004, p. 18)**

   **Formulas needed here:** span length ~ Geometric(p), mean length =
   1/p — state both distributions explicitly with their actual parameter
   values rather than only naming "geometric": masked-span length ~
   Geometric(0.1) (mean 10 characters); unknown-length gap `[#]` ~
   Geometric(0.25) − 1 (mean 3, ~25% of examples draw 0 → no gap that
   example). **(LacunaBERT-slav model card, "Masking")**
7. Decoding is non-autoregressive and iterative by default: the model
   attends bidirectionally to the whole sequence, greedily fills whichever
   open mask (or resolves whichever open gap-length decision) it is
   currently most confident about, then re-runs the full forward pass so
   later fills condition on earlier ones. A beam-search decoding variant —
   branching every open position each iteration, length-normalized scoring
   — mirrors Aeneas's non-sequential beam search directly and is already
   implemented, not a proposed extension. **(LacunaBERT-slav model card,
   "Architecture"; repo: `KyivanRestorer`/`KyivanBeamRestorer`,
   `src/model/inference.py` / `inference_beam.py`)** The headline CER
   numbers in Results are reported at `beam_width=1` (i.e. the greedy
   case) — re-running with beam search enabled is a direct, already-coded
   experiment, not future engineering work.
8. Loss weights: restore 5.0, unk 1.0 — restore dominates by design
   (it's the primary task); both also started as literal copies of
   Aeneas's Latin configuration.

**Formula needed here (closes out Methods — the one joint-training
equation the paper needs, ties Date/Dialect/Text restoration back
together):**

   L = 5·L_restore + 1·L_unk + 0.5·L_date + 0.5·L_region

   State each term's own loss type alongside it when this goes in the
   draft — restore/unk/region are cross-entropy (label smoothing 0.05 on
   restore-adjacent per an earlier prep pass, verify that's still current
   for this checkpoint before citing it), date is KL-divergence against
   the soft 20-bin target distribution, no smoothing applicable to a
   KL term.

---

## Results

**Table needed at the top of this section:** the headline numbers, one
table, two columns (Test B official / periodic training `eval`) × rows
(restoration Hit@1, Hit@5, date MAE, date accuracy, date macro-F1, region
accuracy, region macro-F1) — points 1–2 below are the prose walkthrough of
exactly this table, built once so the two columns sit side by side instead
of being compared across two separate paragraphs.

1. **Official, published numbers (Test B — real editorial lacunae):**
   restoration Hit@1 = **0.2567**, Hit@5 = **0.6044**; date classification
   MAE = **84.06 years**, accuracy = **0.2946**, macro-F1 = **0.14**;
   region classification accuracy = **0.835**, macro-F1 = **0.5652**.
   **(LacunaBERT-slav model card, "Tasks")** These are the numbers to lead
   with — they're the model's own published record, on the harder,
   real-damage split.
2. **A striking gap vs. the periodic training-time eval split:** during
   training, the same model reached region accuracy 0.977 and date MAE
   43.8 years on the periodic `eval` split (dynamically re-masked,
   distribution-matched to training). **(`training_log_20260723_000903.
   json`, last logged eval, epoch ≈49.9)** Both auxiliary heads look far
   worse on Test B (region 0.835 vs. 0.977; date MAE 84 vs. 44 years) than
   during training — the same "synthetic eval overstates real utility"
   pattern as the restoration task, and arguably a *sharper* instance of
   it than restoration itself. Worth stating as its own point, not folded
   into (3) below — the auxiliary heads degrade more, proportionally, than
   the primary restoration head does.
3. **Test A vs. Test B, restoration specifically:** top-1 accuracy 0.341
   on synthetic masking vs. 0.257 on real editorial lacunae (5,682 vs.
   8,314 scored positions) — an 8-point gap. **(computed from
   `pred_report_test_a/b_20260723_000903.csv`; Test A isn't in the
   published model card, this is separate analysis on the same run)**

   **Table needed here:** two rows (Test A / Test B) × two columns
   (scored positions, top-1 accuracy) — the numbers above, tabulated. Small
   but this is literally the paper's headline comparison; it should be
   readable at a glance, not only stated in a sentence.
4. **Character error rate, the sharpest single result in the paper.**
   Given the gap's true length: CER = 0.7366 (5,165 lacunae; best case,
   length-1 gaps at 51.5% of all real lacunae: CER = 0.6235) — implies
   ≈26% per-character accuracy, consistent with (3). Left to determine the
   length itself via the Unk head: CER = 6.79 overall, degrading further
   on the same majority length-1 case (CER = 14.18) — because each
   length-expansion decision compounds independently, so even a
   respectable per-step accuracy collapses fast over several consecutive
   decisions. **(LacunaBERT-slav model card, "Tasks"; matches Eremeev's
   direct report)** Reported at `beam_width=1` (greedy) — since beam
   search is already implemented (Methods, Text restoration 7), re-running this specific
   number with beam search enabled is the natural, low-effort next
   experiment, and would strengthen this result considerably either way
   (shows the gap is decoding-strategy-fixable, or confirms it's a deeper
   modeling problem).

   **Figure needed here — the strongest visual in the paper:** CER vs.
   true gap length, two lines (known-length / unknown-length), x-axis =
   gap length 1 through ≥20, y-axis = CER (log scale recommended — the
   two lines span 0.6–0.9 vs. 1.7–14.2, a linear axis would flatten the
   known-length line to near-invisible). Eremeev's full by-length
   breakdown (both conditions, every length bucket from 1 to ≥20) is
   already in this conversation's history — pull it from there rather
   than re-requesting it. This single plot makes the compounding-error
   argument visually obvious in a way the two headline CER numbers alone
   don't.
5. None of the numbers above are broken down by source. Given the
   imbalance in Data point 5, an aggregate figure mostly reflects
   performance on the non-birchbark majority of the corpus — stated here
   as an open limitation, not a validated birchbark-specific result.

---

## Conclusion

1. Kyivan shows that the architectural choices Aeneas validated for Latin
   epigraphy — RoPE, character-only input, tied projection, joint
   multi-task training — transfer to a lower-resource, dialectally
   fragmented Slavic domain, restoring text meaningfully better than
   chance.
2. Real editorial reconstructions are a harder, more informative target
   than synthetic masking, and this shows up most sharply in the auxiliary
   date/region heads, not just restoration (Results points 2–3) — the
   paper's evidence that synthetic/training-distribution-only evaluation
   would have substantially overstated practical utility across every
   task the model performs, not only the headline one.
3. The clearest weakness is length estimation for gaps of unknown size,
   where per-step errors compound (Results point 4). Unlike a typical
   "add beam search as future work" conclusion, Kyivan already has a
   working beam-search decoder (Methods, Text restoration 7) — the immediate next step is
   re-measuring CER with it enabled, not building it. State the outcome
   plainly once that number exists, rather than assuming search alone
   closes the gap.
4. State plainly which specific design choices measurably helped (RoPE
   vs. not, loss-weight tuning) once ablations exist — not yet run as of
   this writing, don't claim it before you have the numbers.
