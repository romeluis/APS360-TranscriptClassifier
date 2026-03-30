"""Token-level noise augmentation for NER training data.

Simulates common PDF text-extraction artifacts so the model generalises
beyond the specific noise pattern of WeasyPrint + pypdf.

All augmentations operate on (tokens, labels) pairs *after* BIO labelling
and preserve label consistency via a repair pass at the end.
"""

import random
import string

# Junk fragments that mimic interleaved page headers / footers
_JUNK_FRAGMENTS = [
    "Page", "Continued", "Unofficial", "Official", "Transcript",
    "CONFIDENTIAL", "—", "...", "–", "|",
]


def _repair_bio(labels):
    """Fix orphaned I- tags by promoting them to B-."""
    repaired = list(labels)
    for i, label in enumerate(repaired):
        if label.startswith("I-"):
            entity_type = label[2:]
            if i == 0 or repaired[i - 1] not in (f"B-{entity_type}", f"I-{entity_type}"):
                repaired[i] = f"B-{entity_type}"
    return repaired


def _merge_tokens(tokens, labels, rng, rate=0.03):
    """Merge adjacent token pairs (simulates missing whitespace)."""
    if len(tokens) < 2:
        return tokens, labels
    out_tokens, out_labels = [], []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and rng.random() < rate:
            out_tokens.append(tokens[i] + tokens[i + 1])
            out_labels.append(labels[i])  # first token's label wins
            i += 2
        else:
            out_tokens.append(tokens[i])
            out_labels.append(labels[i])
            i += 1
    return out_tokens, out_labels


def _split_tokens(tokens, labels, rng, rate=0.03):
    """Split tokens at a random position (simulates extra whitespace)."""
    out_tokens, out_labels = [], []
    for tok, lab in zip(tokens, labels):
        if len(tok) > 2 and rng.random() < rate:
            pos = rng.randint(1, len(tok) - 1)
            out_tokens.append(tok[:pos])
            out_labels.append(lab)
            part2_label = lab
            if lab.startswith("B-"):
                part2_label = "I-" + lab[2:]
            out_tokens.append(tok[pos:])
            out_labels.append(part2_label)
        else:
            out_tokens.append(tok)
            out_labels.append(lab)
    return out_tokens, out_labels


def _insert_junk(tokens, labels, rng, rate=0.02):
    """Insert junk tokens between existing tokens (simulates page headers/footers)."""
    out_tokens, out_labels = [], []
    for tok, lab in zip(tokens, labels):
        if rng.random() < rate:
            # Insert 1–2 junk tokens before this token
            n_junk = rng.randint(1, 2)
            for _ in range(n_junk):
                junk = rng.choice(_JUNK_FRAGMENTS)
                # Sometimes add a page number
                if rng.random() < 0.3:
                    junk = str(rng.randint(1, 20))
                out_tokens.append(junk)
                out_labels.append("O")
        out_tokens.append(tok)
        out_labels.append(lab)
    return out_tokens, out_labels


def _char_typos(tokens, labels, rng, rate=0.02):
    """Introduce character-level typos (swap, drop, double).

    Only applied to O and NAME tokens — CODE, GRADE, SEM have precise
    lexical patterns the model must learn to recognise exactly.
    """
    safe_prefixes = ("B-CODE", "I-CODE", "B-GRADE", "I-GRADE", "B-SEM", "I-SEM")
    out_tokens = []
    for tok, lab in zip(tokens, labels):
        if lab in safe_prefixes or len(tok) < 2 or rng.random() >= rate:
            out_tokens.append(tok)
            continue
        chars = list(tok)
        op = rng.choice(["swap", "drop", "double"])
        pos = rng.randint(0, len(chars) - 1)
        if op == "swap" and pos + 1 < len(chars):
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif op == "drop":
            chars.pop(pos)
        elif op == "double":
            chars.insert(pos, chars[pos])
        out_tokens.append("".join(chars))
    return out_tokens, labels  # labels unchanged


def _delete_o_tokens(tokens, labels, rng, rate=0.01):
    """Delete random O-labelled tokens (simulates missing text)."""
    out_tokens, out_labels = [], []
    for tok, lab in zip(tokens, labels):
        if lab == "O" and rng.random() < rate:
            continue  # skip this token
        out_tokens.append(tok)
        out_labels.append(lab)
    return out_tokens, out_labels


def augment_tokens(tokens, labels, rng=None, intensity=1.0):
    """Apply all noise augmentations to a (tokens, labels) pair.

    Args:
        tokens: list[str]
        labels: list[str] — BIO labels
        rng: random.Random instance (or None for default)
        intensity: float — scales all augmentation probabilities (0 = off, 1 = default)

    Returns:
        (augmented_tokens, augmented_labels)
    """
    if rng is None:
        rng = random.Random()

    def scaled(rate):
        return rate * intensity

    t, l = tokens, labels
    t, l = _merge_tokens(t, l, rng, scaled(0.03))
    t, l = _split_tokens(t, l, rng, scaled(0.03))
    t, l = _insert_junk(t, l, rng, scaled(0.02))
    t, l = _char_typos(t, l, rng, scaled(0.02))
    t, l = _delete_o_tokens(t, l, rng, scaled(0.01))

    # Repair any BIO inconsistencies introduced by augmentation
    l = _repair_bio(l)

    return t, l
