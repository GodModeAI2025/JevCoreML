#!/usr/bin/env python3
"""Breiter Suchlauf fuer die laya-Tokenizer.

Die handverlesenen Faelle haben genau zwei Fehler gefunden, und beide lagen nicht im Algorithmus,
sondern darin, wie Swift und Foundation mit Zeichenketten umgehen: ein fuehrendes U+FEFF
verschwindet beim Dekodieren, und zwei kanonisch gleichwertige Zeichen werden derselbe
Woerterbuchschluessel. Beide Fehler treffen nur einzelne Token des Wortschatzes. Wer sie finden
will, muss den Wortschatz selbst absuchen, nicht sich Saetze ausdenken.

Deshalb kommen die Faelle hier aus drei Quellen:
  1. jedes Token des Wortschatzes, das ein kanonisch gleichwertiges Gegenstueck hat
  2. jedes Token, das mit U+FEFF oder einer kombinierenden Marke beginnt
  3. dazu der uebliche Mix aus Basiszeichen und Marken in unsortierter Reihenfolge
"""
import argparse
import itertools
import unicodedata

import layax

BASES = ["a", "A", "e", "o", "n", "u",
         "é", "А", "א", "ا", "ก",
         "一", "あ", "가", "ᄀ",
         "\U000200C5", "\U00010415", "\U00020000", "\U0001F600", "\U0001D158"]

MARKS = ["̀", "́", "̂", "̈", "̊", "̧", "̨",
         "̴", "ͅ", "ְ", "ּ", "ً", "ٓ", "़",
         "ั", "ཱ", "ི", "⃐", "⃠", "゙", "〪",
         "\U000101FD", "\U00010376", "\U0001D165", "\U0001D16D", "\U0001D1AA"]


def vocabulary_traps(tok):
    """Token, bei denen die Sprachlaufzeit des Ports stolpern kann."""
    vocab = tok.get_vocab()
    by_nfc = {}
    for token in vocab:
        by_nfc.setdefault(unicodedata.normalize("NFC", token), []).append(token)
    out = []
    for group in by_nfc.values():
        if len(group) > 1:
            # Kanonisch gleichwertig, aber verschiedene Token. Genau hier fielen im Port
            # U+0341 und U+0301 zusammen.
            out.extend(group)
    for token in vocab:
        if token.startswith("﻿") or (token and unicodedata.combining(token[0])):
            out.append(token)
    # Ohne die Metaspace-Marke, die ist kein Text, sondern deren Kodierung.
    return sorted({t.replace("▁", " ") for t in out if t.strip("▁")})


def main():
    layax.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=list(layax.CHECKPOINTS), default="english")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    agent = layax.load_laya(args.checkpoint)
    tok = agent.tok
    out = args.out or str(layax.ROOT / "Golden" / "laya" / args.checkpoint / "tokenizer-fuzz.json")

    texts = list(vocabulary_traps(tok))
    traps = len(texts)
    for base in BASES:
        for m1, m2 in itertools.permutations(MARKS, 2):
            texts.append(base + m1 + m2)
        for m in MARKS:
            texts.append(base + m)
            texts.append(base + m + m)
            texts.append(m + base)
    for m1, m2, m3 in itertools.islice(itertools.permutations(MARKS, 3), 400):
        texts.append("x" + m1 + m2 + m3)
    # Und dieselben Faelle noch einmal eingebettet, damit auch der Weg durch die Zerlegung
    # in Stuecke geprueft wird und nicht nur der Sonderfall "Text besteht aus einem Zeichen".
    texts.extend("Vor " + t + " nach" for t in texts[:traps])
    texts = list(dict.fromkeys(texts))

    cases = [{"text": t, "ids": [int(i) for i in tok(t, add_special_tokens=False)["input_ids"]]}
             for t in texts]
    layax.write_json(out, {"count": len(cases), "vocabulary_traps": traps, "cases": cases})
    print(f"{len(cases)} Faelle ({traps} aus dem Wortschatz) nach {out}")


if __name__ == "__main__":
    main()
