#!/usr/bin/env python3
"""Breiter Suchlauf fuer den Tokenizer, jenseits der handverlesenen Faelle.

Der Golden-Korpus in tokenizer.json enthaelt die Faelle, von denen man weiss, dass sie
schwierig sind. Diese Datei sucht nach denen, von denen man es nicht weiss: Kombinationen aus
Basiszeichen und mehreren Marken in kanonisch unsortierter Reihenfolge, ueber alle Ebenen.

Genau dort wird die Abweichung vermutet: ICU kennt die kanonische Kombinationsklasse vieler
Zeichen, die der Referenz-Normalizer als Startzeichen fuehrt. Wer da falsch liegt, sortiert um.
"""
import argparse, itertools, json
import jevx
from kev.model import load_tokenizer, user_tokens

BASES = ["a", "A", "e", "o", "n", "u",
         "é", "А", "א", "ا", "ก",
         "一", "あ", "가", "ᄀ",
         "\U000200C5", "\U00010415", "\U00020000", "\U0001F600", "\U0001D158"]

# Marken quer durch die Klassen: Umschliessende, Ueber-, Unter-, Vokalzeichen, Ebenen.
MARKS = ["̀", "́", "̂", "̈", "̊", "̧", "̨",
         "̴", "ͅ", "ְ", "ּ", "ً", "ٓ", "़",
         "ั", "ཱ", "ི", "⃐", "⃠", "゙", "〪",
         "\U000101FD", "\U00010376", "\U0001D165", "\U0001D16D", "\U0001D1AA"]


def main():
    jevx.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jevx.ROOT / "Golden" / "tokenizer-fuzz.json"))
    a = ap.parse_args()
    tok = load_tokenizer("Qwen/Qwen3-0.6B-Base", revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")

    texts = []
    for base in BASES:
        for m1, m2 in itertools.permutations(MARKS, 2):
            texts.append(base + m1 + m2)
        for m in MARKS:
            texts.append(base + m)
            texts.append(base + m + m)
            texts.append(m + base)
    # Dazu Marken ohne Basis und drei hintereinander.
    for m1, m2, m3 in itertools.islice(itertools.permutations(MARKS, 3), 400):
        texts.append("x" + m1 + m2 + m3)
    texts = list(dict.fromkeys(texts))

    cases = [{"text": t, "ids": [int(i) for i in tok(t, add_special_tokens=False).input_ids],
              "user_ids": [int(i) for i in user_tokens(tok, t)]} for t in texts]
    jevx.write_json(a.out, {"count": len(cases), "cases": cases})
    print(f"{len(cases)} Faelle nach {a.out}")


if __name__ == "__main__":
    main()
