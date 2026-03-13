# Address Error Correction with Vendor Name

## Problem

We have receipt images processed by OCR. We know the vendor name and have an
OCR'd address. The address frequently contains errors:

```
OCR output:  avenue de e1'quest
Correct:     avenue de l'ouest
```

Common OCR corruptions include `l`→`1`, `O`→`0`, `rn`→`m`, dropped diacritics
(`é`→`e`), merged/split tokens, and dropped small words (French articles like
"de", "la"). We need to correct these addresses using labeled streetmap data
(Overture Maps, OpenAddresses) with vendor names as an anchor.

**Requirements:** No API calls at serve time. No licensing fees. France,
Germany, US, Canada. Lookup under 250ms, ideally under 50ms.

---

## Critical Insight: Vendor Name Is the Primary Signal

All three approaches below benefit enormously from the same strategy Google
uses: **use the vendor name to narrow the candidate set first, then
fuzzy-match the address second.** If your reference database has 50 million
addresses, but only 3 locations for "Boulangerie Dupont", your address matcher
only needs to pick between 3 candidates — not 50 million. This turns a hard
fuzzy-matching problem into a trivial one.

**Data source for vendor→address pairs:** Overture Maps Foundation Places theme
(64M+ POIs globally, includes brand/name fields + structured addresses, CDLA
Permissive 2.0 license, downloadable as GeoParquet from S3). Supplement with
OpenAddresses (600M+ addresses from government sources) for address-only
validation.

---

## Option 1: Libpostal + Elasticsearch with N-Gram Tokenizers

### How it works

1. **Index time:** Download Overture Maps Places for your four countries. Run
   each address through libpostal's `expand_address` (produces normalized
   canonical forms — expands abbreviations, handles multilingual conventions,
   normalizes numerics). Index vendor name + all address expansions into
   Elasticsearch using a **custom n-gram analyzer** (character n-grams of size
   3-4) alongside a standard analyzer.

2. **Query time:** Run the OCR'd address through libpostal `parse_address` to
   extract components (street, house number, postcode, city). Run
   `expand_address` on each component. Build an Elasticsearch query that:
   - Filters by vendor name (using `match` with `fuzziness: "AUTO"`)
   - Boosts exact postcode matches
   - Searches street name using the n-gram field with a `minimum_should_match`
     threshold
   - Falls back progressively: full match → street+city → street only

3. **Scoring:** Elasticsearch's BM25 ranking + field boosts. Return the
   top candidate if it exceeds a confidence threshold; otherwise return
   "no correction."

### Pros

- **Battle-tested architecture.** This is essentially the Pelias geocoder's
  design (used by Mapzen, now maintained by the community). Well-documented,
  widely deployed.
- **Libpostal handles multilingual normalization natively.** "Straße"/"Str.",
  "Rue"/"R.", "Boulevard"/"Bd", "Avenue"/"Ave" — all resolved before matching.
  Handles French vigesimal numbers ("quatre-vingt-douze" → "92"), German
  compounds ("Rosenstrasse" = "Rosen Strasse"), and 60+ languages.
- **N-gram tokenizers are pre-computed at index time.** Query speed is
  consistent regardless of data size. Sub-20ms queries on well-indexed data.
- **Elasticsearch scales well.** Handles tens of millions of records without
  difficulty. Supports field-specific boosting, custom analyzers per field, and
  complex multi-field queries.
- **Libpostal's parser is highly accurate.** 99.45% correct full parses on a
  global test set. The CRF was trained on 1 billion+ addresses.

### Cons

- **Libpostal cannot correct OCR errors.** It is dictionary-based, not fuzzy.
  `e1'quest` passes through unchanged because it matches no known abbreviation.
  The n-gram tokenizer in Elasticsearch must do all the heavy lifting for
  character-level corruption. This means the architecture has a blind spot:
  libpostal normalizes the *structure*, but the *character-level errors* are
  only handled by the ES n-gram overlap, which has no OCR-specific awareness.
- **Elasticsearch fuzzy queries cap at edit distance 2.** For severely
  corrupted tokens like `e1'quest` → `l'ouest` (edit distance 4+), the `fuzzy`
  query will miss entirely. You must rely on the n-gram analyzer path, which is
  less precise for ranking.
- **Operational complexity.** Requires running Elasticsearch (or OpenSearch) as
  a service alongside your application. JVM tuning, index management, cluster
  health monitoring. Not a simple library you can embed.
- **Libpostal memory footprint is large.** ~2GB on disk, reported 4-6GB RAM in
  production. Must run as a separate service (typically the Go-based
  libpostal-service).
- **No OCR-aware character confusion.** The system treats `1` and `l` as
  completely unrelated characters. It has no model of which characters OCR
  commonly confuses.

### Expected accuracy on your example

`"avenue de e1'quest"` → libpostal normalizes `avenue` but leaves `e1'quest`
unchanged → ES n-gram query on `e1'quest` shares trigrams `"e1'", "1'q",
"'qu", "que", "ues", "est"` with `l'ouest` which shares `"l'o", "'ou", "oue",
"ues", "est"` → only 2 of 6 trigrams overlap → **marginal match, likely
below threshold.** The vendor name filter would need to narrow candidates to
1-3 records for this to succeed.

---

## Option 2: OCR Confusion Matrix + Phonetic Normalization + Trigram Matching

### How it works

This approach adds an explicit OCR error correction layer before any address
parsing. It models *how OCR fails* rather than relying on generic fuzzy
matching.

1. **OCR confusion matrix pre-processing.** Before any address parsing, apply
   a character-level confusion matrix that maps common OCR substitutions:
   - `1` → also consider `l`, `I`, `i`
   - `0` → also consider `O`, `o`
   - `rn` → also consider `m`
   - `|` → also consider `l`, `I`, `1`
   - `'` → also consider `'`, `'`, `` ` ``
   - Drop/restore diacritics: `e` ↔ `é`/`è`/`ê`, `a` ↔ `à`/`â`, `u` ↔ `ü`/`ù`/`û`
   - `B` ↔ `ß` (German)

   For each OCR'd token, generate candidate corrections by applying the
   confusion matrix (limited depth to keep candidate count bounded). This is
   similar to Peter Norvig's spell-checker but with an OCR-specific edit model
   instead of uniform edit distance.

2. **Phonetic normalization.** Apply a phonetic algorithm suited to the
   receipt's language:
   - French: Soundex français or phonetic folding rules
   - German: Kölner Phonetik (Cologne phonetic)
   - English: Double Metaphone

   This catches errors where OCR produces a visually similar but phonetically
   identical string ("ouest" / "quest" sound different, but "Straße" / "Strasse"
   are phonetically identical).

3. **Libpostal normalization.** Run the corrected candidates through libpostal
   `expand_address` + `parse_address` to get structured components.

4. **Index and match in PostgreSQL with pg_trgm.** Store reference data
   (Overture Places + OpenAddresses) in PostgreSQL. Create GIN trigram indexes
   on vendor_name, street, city, and postcode columns. Query:
   ```sql
   SELECT *, similarity(street, $query_street) AS street_sim,
             similarity(vendor_name, $query_vendor) AS vendor_sim
   FROM addresses
   WHERE vendor_name % $query_vendor   -- trigram similarity filter
     AND (postcode = $query_postcode OR city % $query_city)
   ORDER BY vendor_sim * 0.4 + street_sim * 0.4 +
            (postcode = $query_postcode)::int * 0.2 DESC
   LIMIT 1;
   ```

### Pros

- **Directly models the OCR error source.** Instead of hoping generic fuzzy
  matching catches `1`→`l`, this approach knows that `1` and `l` are commonly
  confused by OCR and explicitly generates the correction. This is the single
  most effective intervention for your specific problem.
- **Phonetic matching catches a class of errors that edit distance misses.**
  OCR might produce a string that is far in edit distance but phonetically
  close (or vice versa). Combining phonetic codes with character-level
  correction covers both failure modes.
- **PostgreSQL-only stack.** No Elasticsearch, no JVM. pg_trgm is a built-in
  extension. Much simpler to deploy and operate.
- **pg_trgm is fast for your data size.** With GIN indexes on a few million
  rows, queries return in 5-20ms. The full pipeline (confusion matrix →
  phonetic → libpostal → pg_trgm query) should land under 50ms.
- **Diacritics restoration is explicit.** French and German addresses are
  heavily diacriticked. By modeling diacritics as an OCR confusion class, you
  catch `Dusseldorf` → `Düsseldorf` and `Chatelet` → `Châtelet` systematically.
- **Can be extended incrementally.** Start with a small confusion matrix, tune
  it based on measured OCR errors from your actual receipt corpus.

### Cons

- **Requires building and tuning the confusion matrix.** There's no
  off-the-shelf OCR confusion matrix for receipt text. You need to either:
  (a) analyze a sample of your OCR'd receipts against ground truth to learn
  the error distribution, or (b) use a generic matrix and tune it. This is
  engineering work.
- **Candidate explosion with deep confusion rewrites.** If you allow 3
  substitutions per token and each has 3 candidates, you generate 27 candidate
  strings per token. For a 4-token street name, that's 531K combinations.
  Must be bounded carefully (e.g., max 2 substitutions, prune by dictionary
  lookup).
- **Phonetic algorithms are language-specific.** You need to detect the
  receipt's language (or country) to apply the right phonetic algorithm. Wrong
  phonetic algorithm → wrong normalization. Fortunately, you likely know the
  country from the vendor.
- **pg_trgm degrades on very short strings.** House numbers, postcodes, and
  abbreviations have few trigrams, making similarity scores noisy. Best to
  match these components exactly or with explicit edit distance rather than
  trigram similarity.
- **No learned ranking.** The weighted scoring formula
  (`vendor_sim * 0.4 + street_sim * 0.4 + ...`) is hand-tuned. It works
  but doesn't learn from feedback. If your miss rate is high, there's no
  automatic way to improve scoring — you retune weights manually.

### Expected accuracy on your example

`"avenue de e1'quest"` → confusion matrix rewrites `e1` → `el`, `1'` → `l'`,
producing candidate `"avenue de el'quest"` and `"avenue de l'quest"` among
others → libpostal normalizes `avenue` → pg_trgm on `l'quest` vs `l'ouest`
(similarity ~0.55, above the default 0.3 threshold) → **match found,**
especially when combined with vendor name narrowing. This approach handles
your specific example significantly better than Option 1.

---

## Option 3: Character N-Gram Embeddings + Dense Retrieval

### How it works

This approach treats address matching as a **retrieval problem** and uses
learned or engineered embeddings that are inherently robust to character-level
noise. This is closest to how Google/Meta handle address and entity matching
at scale.

1. **Encode reference addresses as character n-gram vectors.** For each
   (vendor_name, address) pair in Overture Maps, produce a vector
   representation based on overlapping character n-grams (3-grams, 4-grams,
   5-grams). Two approaches:

   **a) TF-IDF on character n-grams (no ML).** Concatenate vendor name +
   address, extract all character n-grams, weight by TF-IDF. Store as sparse
   vectors. This is Facebook's original FastText-like approach for entity
   matching. Sparse dot product gives a similarity score that is inherently
   robust to character-level noise because most n-grams survive even when a
   few characters are corrupted.

   **b) Learned dense embeddings.** Train a small model (e.g., a character-CNN
   or a lightweight transformer) on (corrupted_address, correct_address) pairs
   to produce fixed-size dense vectors. Synthetic training data can be
   generated by applying OCR-like corruptions to known-good addresses.

2. **Index vectors in a vector database.** Store embeddings in FAISS (Meta's
   library, runs locally, no server needed) or Hnswlib. For sparse TF-IDF
   vectors, use sparse FAISS indexes or Elasticsearch's dense_vector field.

3. **Two-stage retrieval at query time:**
   - **Stage 1 (retrieval):** Encode the OCR'd (vendor_name, address) into the
     same vector space. Retrieve top-K (e.g., K=20) nearest neighbors by
     approximate nearest neighbor search. This is sub-10ms on FAISS with
     millions of vectors.
   - **Stage 2 (reranking):** Score the K candidates using a combination of:
     - Character-level Jaro-Winkler distance on individual components
       (street, city, postcode) after libpostal parsing
     - Exact postcode match bonus
     - Vendor name similarity (Jaro-Winkler or token overlap)

     Return the top candidate if it exceeds a confidence threshold.

### Pros

- **Most robust to severe OCR corruption.** Character n-gram representations
  are designed to survive exactly the kind of noise OCR produces. `"avenue de
  e1'quest"` and `"avenue de l'ouest"` share the vast majority of their
  character 4-grams: `"aven"`, `"venu"`, `"enue"`, `"nue "`, `"ue d"`,
  `"e de"`, `" de "` — 7 of the first 7 n-grams are identical. Even the
  corrupted tail shares `"ques"`, `"uest"`. The cosine similarity would be
  high (~0.75+) despite the edit distance being 4+.
- **No hand-tuned confusion matrix needed.** The character n-gram
  representation is inherently robust to substitutions, insertions, and
  deletions without needing to enumerate specific OCR error patterns. If you
  use the learned embedding variant, the model discovers the error patterns
  from training data automatically.
- **FAISS is extremely fast and runs locally.** For 5-10 million vectors of
  dimension 128-256, FAISS HNSW index returns top-20 results in 1-5ms on CPU.
  No external service needed — it's a C++ library with Python bindings.
- **Scales gracefully.** Adding more countries or data sources means adding
  more vectors. No schema changes, no reindexing, no analyzer reconfiguration.
- **The two-stage architecture allows precision tuning.** Stage 1 (vector
  retrieval) optimizes for recall — don't miss any plausible candidate. Stage 2
  (reranking) optimizes for precision — pick the right one. These can be tuned
  independently.
- **Vendor name naturally weights the embedding.** Because vendor name is
  prepended to the address in the embedding input, candidates from the right
  vendor are pulled closer in vector space. Even if the address is severely
  garbled, a matching vendor name provides strong signal.

### Cons

- **TF-IDF variant produces high-dimensional sparse vectors.** Character
  n-gram vocabularies are large (26^4 ≈ 460K possible 4-grams). Sparse
  vectors work but require more memory than dense vectors. FAISS supports
  this via `IndexIVFFlat` with product quantization, but it's more complex
  to tune.
- **Learned embedding variant requires training data.** You need (corrupted,
  correct) address pairs. These can be synthesized by applying random OCR-like
  noise to Overture addresses, but the quality of the synthetic noise model
  matters — if your synthetic errors don't match real OCR errors, the model
  won't generalize well.
- **Vector similarity is not structurally aware.** Cosine similarity on
  character n-grams doesn't know that "14 Rue de Rivoli" and "41 Rue de
  Rivoli" are different addresses (house numbers transposed). The reranking
  stage must catch this, which means you still need libpostal for component
  parsing in stage 2.
- **More complex architecture than Options 1 or 2.** You're building a
  two-stage retrieval pipeline: embedding generation → ANN index → reranker.
  This is more engineering than a single Elasticsearch or PostgreSQL query.
- **FAISS index must fit in memory.** For 10 million vectors at 256
  dimensions (float32), that's ~10GB RAM. Manageable but non-trivial.
  Product quantization can reduce this 4-8x at the cost of some recall.
- **Harder to debug.** When a match fails, it's not obvious why — you can't
  easily inspect why two vectors are far apart. With trigram matching, you can
  see exactly which trigrams overlapped. With embeddings, the failure mode is
  opaque.

### Expected accuracy on your example

`"Boulangerie Dupont avenue de e1'quest"` → character 4-grams include
`"Boul"`, `"oula"`, `"ulan"`, `"lang"`, `"ange"`, `"nger"`, `"geri"`,
`"erie"`, `"rie "`, ..., `"aven"`, `"venu"`, `"enue"`, ..., `"uest"` →
reference `"Boulangerie Dupont avenue de l'ouest"` shares ~85% of 4-grams →
**cosine similarity ~0.85, strong match.** Reranking confirms via vendor name
exact match + postcode check → **correct address returned with high
confidence.**

---

## Comparison Summary

| Factor | Option 1: Libpostal + ES N-grams | Option 2: OCR Confusion + pg_trgm | Option 3: Char N-gram Embeddings |
|--------|----------------------------------|-----------------------------------|----------------------------------|
| **Handles `e1'quest`→`l'ouest`** | Marginal (low trigram overlap) | Yes (confusion matrix rewrites `1`→`l`) | Yes (n-gram overlap is high) |
| **Handles dropped diacritics** | Partial (libpostal strips some) | Yes (explicit diacritics layer) | Yes (n-grams mostly survive) |
| **Handles abbreviation variants** | Excellent (libpostal's strength) | Good (libpostal + phonetic) | Moderate (no structural awareness) |
| **Vendor name leverage** | ES field boost | SQL WHERE clause | Embedded in vector naturally |
| **Query latency** | 10-30ms | 10-30ms | 2-10ms retrieval + 5ms rerank |
| **Infrastructure** | Elasticsearch + libpostal service | PostgreSQL only (+ libpostal) | FAISS (in-process) + libpostal |
| **Operational complexity** | High (JVM, ES cluster) | Low (single Postgres instance) | Medium (index build, memory) |
| **Tuning effort** | Moderate (analyzer config, boosts) | High (confusion matrix, weights) | Moderate-High (embeddings, reranker) |
| **Debuggability** | Good (ES explain API) | Best (SQL is transparent) | Worst (opaque vector distances) |
| **Memory** | ES heap + libpostal (~8GB+) | Postgres shared_buffers + libpostal (~6GB) | FAISS index + libpostal (~12-16GB) |

---

## Recommendation

For your specific problem — OCR-corrupted addresses on receipts with known
vendor names — **Option 2 (OCR Confusion Matrix + pg_trgm) is the best
starting point.** It directly models the error source, is the simplest to
deploy, and gives you the most transparent debugging. The vendor name
narrowing via a SQL WHERE clause is trivially fast and reduces the fuzzy
matching problem to a handful of candidates.

If Option 2's miss rate is too high on severe corruption (measured on a
real sample of your receipts), **add Option 3's embedding retrieval as a
fallback path.** Use Option 2 as the fast primary path and fall back to
vector search only when Option 2 returns no match above threshold. This
hybrid avoids the complexity of running vector search for every query while
catching the long tail of severe errors.

Option 1 is the most commonly documented architecture but is actually the
worst fit for your specific problem because neither libpostal nor
Elasticsearch's fuzzy queries handle the degree of OCR corruption you're
seeing.
