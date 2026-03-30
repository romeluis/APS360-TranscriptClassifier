# Implementation Plan: Close Train/Test Distribution Gap via PDF-Extracted Training Data

## Problem Summary

The BERT NER model trains on perfectly clean HTML-extracted tokens but runs inference on noisy PDF-extracted tokens. This distribution mismatch causes the model to perform no better than the rule-based baseline on real PDFs despite 98.6% val F1.

## Architecture Overview

The fix requires changes at three levels:
1. **Generator pipeline** — add a PDF-round-trip extraction mode alongside the existing HTML extraction
2. **Alignment module** — new module to map known entity annotations onto noisy PDF text
3. **Noise augmentation** — optional token-level perturbations during training for additional robustness

The JSON output format (`{"tokens": [...], "ner_tags": [...], "metadata": {...}}`) remains unchanged, preserving backward compatibility with the model, evaluation, and data loading code.

---

## Phase 1: PDF Text Alignment Module (New File)

### File: `transcript_generator/pdf_aligner.py` (NEW)

This is the core new module. It takes the known entity annotations from the HTML pipeline and transfers them onto the noisy text extracted from the rendered PDF.

#### Algorithm: Character-Level Sequence Alignment (Option A from requirements)

**Why Option A over Option B (entity-value matching):**
- Option B (searching for entity values in PDF text) is fragile when PDF text reorders content (e.g., two-column layouts where pypdf interleaves columns). A course code appearing in a header row vs. a data row is indistinguishable.
- Option B also risks false positives: "Fall" appearing in "Niagara Falls" or "B" appearing in non-grade contexts.
- Option A uses the structural correspondence between the HTML source text and the PDF text. Even when PDF extraction reorders blocks, within each block the character sequence is preserved, and `difflib.SequenceMatcher` will find the longest common subsequences.

**Algorithm steps:**

```
1. From HTML: call existing extract_text_and_annotations(html_string)
   -> (clean_text, annotations)  where annotations = [(start, end, entity_type), ...]

2. From PDF: use pypdf to extract text (same as run_on_pdf.py)
   -> pdf_text

3. Build a character-level entity map on the clean text:
   clean_char_entity[i] = entity_type or None

4. Use difflib.SequenceMatcher(None, clean_text, pdf_text) to get matching blocks:
   Each matching_block = (clean_start, pdf_start, length) means
   clean_text[clean_start:clean_start+length] == pdf_text[pdf_start:pdf_start+length]

5. Transfer annotations through the alignment:
   For each matching_block, for each character position within it:
     pdf_char_entity[pdf_start + offset] = clean_char_entity[clean_start + offset]

6. Build token-level BIO labels from pdf_char_entity:
   - Whitespace-tokenize pdf_text (same as run_on_pdf.py: text.split())
   - For each token, determine entity from character spans (majority vote, same logic as existing text_to_bio_labels)
   - Assign B-/I- prefix based on sequence continuity

7. Return (pdf_tokens, pdf_labels)
```

**Key design decisions:**
- Use `difflib.SequenceMatcher` with `autojunk=False` to avoid discarding common characters as junk. Character-level matching is more precise than word-level.
- Set `autojunk=False` because some entity characters (digits, common letters) would otherwise be treated as junk.
- After alignment, any PDF character that did NOT map to a clean-text character gets entity=None (tagged O). This safely handles PDF artifacts like page numbers, repeated headers, etc.

**Quality gate:** After generating labels, compute alignment coverage = (number of entity characters successfully transferred) / (total entity characters in clean text). If coverage < 80%, log a warning. If < 50%, skip this sample and fall back to HTML extraction, logging the failure.

#### Functions in `pdf_aligner.py`:

```python
def align_pdf_to_html(clean_text: str, annotations: list[tuple], pdf_text: str) -> list[tuple]:
    """Transfer character-level annotations from clean HTML text to noisy PDF text.
    Returns: list of (start, end, entity_type) annotations on pdf_text."""

def extract_pdf_labels(html_string: str, pdf_path: str) -> tuple[list[str], list[str], dict]:
    """Full pipeline: HTML annotations + PDF text -> (tokens, bio_labels, alignment_stats).
    alignment_stats includes coverage metrics for quality monitoring."""

def compute_alignment_coverage(clean_annotations, pdf_annotations, clean_text, pdf_text) -> float:
    """Fraction of entity character mass successfully transferred."""
```

This module imports from:
- `label_extractor.extract_text_and_annotations` (existing) — for the clean-text side
- `label_extractor.text_to_bio_labels` (existing) — for converting char annotations to BIO
- `pypdf.PdfReader` — for PDF text extraction (same as `run_on_pdf.py`)
- `difflib.SequenceMatcher` — stdlib, no new dependency

---

## Phase 2: Modify the Generator Pipeline

### File: `transcript_generator/main.py` (MODIFY)

**Changes:**

1. **New CLI flag: `--pdf-extract`**
   - When set, the pipeline renders HTML->PDF->extracts text->aligns labels (the new path)
   - When not set (default), uses existing HTML extraction (backward compatible)
   - Implies PDF rendering (overrides `--no-pdf`)

2. **New CLI flag: `--augment-noise`** (optional, Phase 4)
   - Applies token-level noise augmentation to training data

3. **Modify `generate_for_template()` function:**
   - Add parameter `pdf_extract: bool = False`
   - When `pdf_extract=True`:
     a. Always render PDF (regardless of `--no-pdf`)
     b. Call `pdf_aligner.extract_pdf_labels(html_string, pdf_path)` instead of `label_extractor.extract_labels(html_string)`
     c. If alignment fails quality gate, fall back to HTML extraction and log
   - Store alignment stats in metadata for monitoring

4. **Updated metadata schema** (backward compatible additions):
   ```json
   {
     "tokens": [...],
     "ner_tags": [...],
     "metadata": {
       "template_id": "...",
       "transcript_id": "...",
       "num_semesters": 3,
       "num_courses": 12,
       "extraction_mode": "pdf",        // NEW: "html" or "pdf"
       "alignment_coverage": 0.95,      // NEW: only present when mode="pdf"
       "pdf_token_count": 187,          // NEW
       "html_token_count": 175          // NEW
     }
   }
   ```

5. **Manifest additions:**
   - `extraction_mode` field
   - Aggregated alignment stats (mean/min coverage, fallback count)

### File: `transcript_generator/pdf_renderer.py` (MODIFY - minor)

Currently minimal (3 lines of logic). May need to:
- Return the path or bytes so the caller can pass it to `pdf_aligner`
- Add error handling for WeasyPrint failures (some templates with complex CSS may fail)
- Optionally accept a `tempfile` path for cases where we don't want to persist the PDF

Actually, looking at the current code, `render_pdf` already writes to `output_path` and returns nothing. The caller in `main.py` already has the `pdf_path`. No change needed to `pdf_renderer.py` itself; `pdf_aligner.py` will just read from the same path.

### File: `transcript_generator/label_extractor.py` (NO CHANGES)

The existing module is correct and complete. The new `pdf_aligner.py` imports `extract_text_and_annotations` and `text_to_bio_labels` from it. No modifications needed.

---

## Phase 3: Noise Augmentation Module (New File)

### File: `transcript_generator/noise_augmenter.py` (NEW)

Even with PDF-extracted training data, the model should be robust to variations beyond what WeasyPrint+pypdf produces. Real-world PDFs come from diverse renderers.

**Augmentation types (applied at the token level, after BIO labeling):**

1. **Character-level typos** (low probability, ~2% of tokens):
   - Swap adjacent characters: "CSC301" -> "CS3C01"
   - Drop a character: "Statistics" -> "Staistics"
   - Insert a random character: "Fall" -> "Falll"
   - Only applied to O tokens and NAME tokens (not CODE/GRADE/SEM, which need exact matches)

2. **Token merging** (~3% probability):
   - Merge two adjacent tokens: ["CSC301", "Machine"] -> ["CSC301Machine"]
   - The merged token gets the label of its first constituent (or O if both are O)
   - Simulates missing whitespace in PDF extraction

3. **Token splitting** (~3% probability):
   - Split a token at a random position: "Statistics" -> ["Stat", "istics"]
   - First part keeps the original label, second part gets I- continuation
   - Simulates extra whitespace insertion by PDF extraction

4. **Junk token insertion** (~2% probability):
   - Insert page numbers, header fragments, or random strings between tokens
   - Inserted tokens get O label
   - Simulates page header/footer interleaving

5. **Token deletion** (~1% probability, O tokens only):
   - Remove a random O token
   - Simulates missing text from PDF extraction

**Interface:**
```python
def augment_tokens(tokens: list[str], labels: list[str], rng: random.Random, intensity: float = 1.0) -> tuple[list[str], list[str]]:
    """Apply random noise augmentation. Returns (augmented_tokens, augmented_labels).
    intensity scales all probabilities (0.0 = no augmentation, 1.0 = default, 2.0 = aggressive)."""
```

**BIO consistency repair:** After all augmentations, run a pass to fix any I- tags that lost their preceding B- tag (same logic as `validate_bio_labels` but auto-fixing).

**Integration with main.py:** When `--augment-noise` is passed, apply augmentation after label extraction (whether HTML or PDF mode). This is a separate concern from the extraction mode.

---

## Phase 4: Verification and Testing

### File: `transcript_generator/test_alignment.py` (NEW - test/verification script)

A standalone script that:
1. Takes a template, generates one transcript
2. Extracts labels via both HTML and PDF paths
3. Prints side-by-side comparison of tokens and labels
4. Reports alignment coverage statistics
5. Validates BIO label consistency on both outputs
6. Can be run across all templates to find problematic layouts

```python
def verify_template(template_path: str, count: int = 5) -> dict:
    """Generate transcripts and compare HTML vs PDF extraction.
    Returns: {
        'template': str,
        'mean_coverage': float,
        'failures': int,
        'entity_agreement': float,  # fraction of entities found in both
        'bio_errors': int,
    }"""
```

### Verification Strategy:

1. **Unit-level:** Test `align_pdf_to_html` on synthetic examples with known character mappings
   - Clean text: "Fall 2021 CSC301 Machine Learning A+"
   - Simulated PDF text: "Fall 2021CSC301 Machine Learning A+" (merged space)
   - Verify annotations transfer correctly

2. **Integration-level:** Run `test_alignment.py` on all 53 templates
   - Expect >90% alignment coverage on most templates
   - Identify templates where PDF text order diverges significantly (e.g., multi-column layouts)
   - These edge cases inform whether we need template-level fallback strategies

3. **Training-level:** After generating PDF-extracted training data:
   - Train the model on PDF-extracted data
   - Evaluate on the same test set (using PDF-extracted test data)
   - Compare val F1 (will likely drop from 98.6% since data is noisier)
   - Then evaluate on real PDFs (the actual goal) and compare against baseline

4. **Augmentation validation:** Compare model performance with and without noise augmentation on real PDFs

---

## Implementation Sequence

### Step 1: Create `pdf_aligner.py`
- Implement `align_pdf_to_html` using `difflib.SequenceMatcher`
- Implement `extract_pdf_labels` as the high-level entry point
- Add alignment coverage computation
- **Dependencies:** existing `label_extractor.py`, `pypdf`, `difflib`

### Step 2: Create `test_alignment.py`
- Test alignment on a few templates manually
- Iterate on alignment quality before modifying the generator
- This catches problems early (e.g., templates where pypdf produces wildly different text order)

### Step 3: Modify `main.py`
- Add `--pdf-extract` CLI flag
- Modify `generate_for_template` to support both extraction modes
- Add alignment stats to metadata and manifest
- Ensure backward compatibility (default behavior unchanged)

### Step 4: Create `noise_augmenter.py`
- Implement token-level augmentations
- Add `--augment-noise` CLI flag to `main.py`
- Test that augmented data still has valid BIO labels

### Step 5: Regenerate training data
- Run: `python main.py --all-templates --count 50 --pdf-extract`
- Monitor alignment coverage across all templates
- Investigate any templates with <80% coverage

### Step 6: Add `pypdf` to `requirements.txt`
- Currently only imported in `run_on_pdf.py` without being listed in requirements
- Add: `pypdf>=4.0.0`

### Step 7: Retrain and evaluate
- Train model on PDF-extracted data
- Evaluate on real PDFs
- Compare with baseline and with model trained on HTML-extracted data

---

## Edge Cases and Risks

### 1. Multi-column layouts
Templates like `37_grid_three_col_courses` and `40_twocol_split_semesters` may produce PDF text where pypdf reads columns in unexpected order (e.g., all of column 1, then all of column 2).

**Mitigation:** The character-level alignment approach handles this gracefully because `SequenceMatcher` finds the longest common subsequences regardless of global ordering. Even if the PDF text has "Fall 2021 Spring 2022 CSC301 Machine..." where the HTML had "Fall 2021 CSC301 Machine... Spring 2022...", the individual entity substrings will still align correctly as long as they appear somewhere in the PDF text.

**Fallback:** If alignment coverage drops below 50%, fall back to HTML extraction for that specific sample and log it.

### 2. Running headers/footers in PDFs
WeasyPrint CSS `position: running(...)` creates repeated headers on each page. The HTML extractor already skips these (via `_find_running_classes`), but pypdf will extract them.

**Mitigation:** These characters won't match any entity annotations (they're O in the HTML), so they'll correctly get O labels in the PDF extraction. They add noise tokens (which is actually beneficial for training robustness).

### 3. Ligatures and special characters
Some fonts may cause pypdf to extract ligatures differently (e.g., "fi" ligature extracted as a single character).

**Mitigation:** `SequenceMatcher` handles this because the surrounding characters still match. The affected tokens get slightly different character boundaries but the entity labels transfer correctly via majority vote.

### 4. Empty PDF text
Some WeasyPrint configurations might produce PDFs where pypdf extracts no text (image-based rendering).

**Mitigation:** Check for empty/near-empty PDF text in `extract_pdf_labels` and fall back to HTML extraction.

### 5. Performance
Rendering PDFs with WeasyPrint is significantly slower than HTML-only extraction (~0.5-2s per PDF vs. ~5ms for HTML extraction). For 53 templates x 50 samples = 2,650 transcripts, this adds ~20-80 minutes of generation time.

**Mitigation:** 
- This is a one-time cost at data generation time, not training time
- The `--no-pdf` default remains for quick iteration
- Consider parallelizing with `concurrent.futures` if needed (out of scope for initial implementation)

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `transcript_generator/pdf_aligner.py` | CREATE | Core alignment algorithm: HTML annotations -> PDF text BIO labels |
| `transcript_generator/noise_augmenter.py` | CREATE | Token-level noise augmentation for training robustness |
| `transcript_generator/test_alignment.py` | CREATE | Verification script comparing HTML vs PDF extraction |
| `transcript_generator/main.py` | MODIFY | Add --pdf-extract and --augment-noise flags, update generate_for_template |
| `requirements.txt` | MODIFY | Add pypdf>=4.0.0 |
| `transcript_generator/label_extractor.py` | NO CHANGE | Existing functions reused by pdf_aligner |
| `transcript_generator/pdf_renderer.py` | NO CHANGE | Already renders HTML to PDF; called as-is |
| `transcript_generator/assembler.py` | NO CHANGE | No modifications needed |
| `model/dataset.py` | NO CHANGE | Reads same JSON format |
| `model/train.py` | NO CHANGE | Reads same JSON format |
| `model/config.py` | NO CHANGE | Label scheme unchanged |

---

## Critical Files for Implementation

1. `transcript_generator/pdf_aligner.py` (NEW) — the alignment algorithm
2. `transcript_generator/main.py` (MODIFY) — pipeline integration
3. `transcript_generator/label_extractor.py` (READ) — provides extract_text_and_annotations and text_to_bio_labels
4. `transcript_generator/noise_augmenter.py` (NEW) — augmentation
5. `transcript_generator/test_alignment.py` (NEW) — verification
