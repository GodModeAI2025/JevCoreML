#!/usr/bin/env python3
"""Unicode-Verhalten der Referenz, Codepunkt für Codepunkt, aus HuggingFace tokenizers selbst.

Die Tokenizer-Korpora prüfen ausgesuchte Sätze. Das hat zwei ganze Fehlerklassen übersehen,
weil kein Satz sie traf: ICU kennt Unicode 17, die Referenz beim Regex Unicode 16 und beim NFC
Unicode 9, und Apple gibt einigen privaten Codepunkten eigene Eigenschaften. Diese Datei hält
deshalb nicht Sätze fest, sondern das Verhalten für jeden einzelnen Codepunkt:

  regex_classes  welche Codepunkte Oniguruma unter \\p{L}, \\p{N} und \\s versteht, als Bereiche
  nfc_single     NFC jedes einzelnen Codepunkts, nur wo er sich ändert
  nfc_marks      jede kombinierende Marke zwischen einem Starter und zwei Referenzmarken
  nfc_astral     jeder astrale Codepunkt mit nachfolgendem Akut, nur wo sich etwas ändert
  nfc_extra      bekannte Fallen: Kompositionen über astrale Starter, Hangul, Unicode-16-Paare

Gemessen wird mit tokenizers.normalizers.NFC und Split-Pretokenizern, also mit genau dem Code,
den die Tokenizer benutzen.
"""
import argparse
import json
import unicodedata

from tokenizers import Regex, normalizers, pre_tokenizers

import layax

PLANES = list(range(0x0, 0x40000)) + list(range(0xE0000, 0xE01F0))


def ranges(points):
    out = []
    for cp in points:
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(layax.ROOT / "Golden" / "unicode-reference.json"))
    args = ap.parse_args()

    classes = {}
    for name, pattern in [("L", r"\p{L}"), ("N", r"\p{N}"), ("S", r"\s")]:
        split = pre_tokenizers.Split(Regex(pattern), behavior="removed")
        hits = [cp for cp in range(0x110000)
                if not 0xD800 <= cp <= 0xDFFF and not split.pre_tokenize_str(chr(cp))]
        classes[name] = ranges(hits)
        print(f"Oniguruma {pattern}: {len(hits)} Codepunkte in {len(classes[name])} Bereichen")

    nfc = normalizers.NFC().normalize_str

    single = {}
    for cp in PLANES:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        got = nfc(chr(cp))
        if got != chr(cp):
            single[cp] = [ord(c) for c in got]
    print(f"NFC einzeln: {len(single)} Codepunkte ändern sich")

    # Jede Marke, die Python kennt (Unicode 15), zwischen U+4E00 und zwei Referenzmarken mit anderer
    # Klasse. Kennt die Referenz die Marke, sortiert sie um, sonst nicht.
    marks = []
    for cp in PLANES:
        if 0xD800 <= cp <= 0xDFFF or unicodedata.combining(chr(cp)) == 0:
            continue
        for text in ["\u4e00" + chr(cp) + "\u0334", "\u4e00\u0301" + chr(cp),
                     "a" + chr(cp) + "\u0316\u0301", "e\u0301" + chr(cp)]:
            marks.append([[ord(c) for c in text], [ord(c) for c in nfc(text)]])
    print(f"NFC Marken: {len(marks)} Proben")

    astral = {}
    for cp in PLANES:
        if cp < 0x10000:
            continue
        text = chr(cp) + "\u0301"
        got = nfc(text)
        if got != text:
            astral[cp] = [ord(c) for c in got]
    print(f"NFC astral mit Akut: {len(astral)} ändern sich")

    # Vorkomponierte Zeichen mit nachfolgender Marke: NFC muss sie zerlegen, umsortieren und neu
    # zusammensetzen. Und jede Hangul-LV-Silbe mit jedem Schluss-Jamo: Foundation komponiert die
    # nicht weiter, wenn die Silbe schon zusammengesetzt ankommt.
    precomposed = []
    for cp in PLANES:
        if 0xD800 <= cp <= 0xDFFF:
            continue
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        for mark in ("\u0323", "\u0301", "\u0334"):
            text = chr(cp) + mark
            got = nfc(text)
            precomposed.append([[ord(c) for c in text], [ord(c) for c in got]])
    for lv in range(0xAC00, 0xD7A4, 28):
        for t in range(0x11A7, 0x11C3):
            text = chr(lv) + chr(t)
            precomposed.append([[ord(c) for c in text], [ord(c) for c in nfc(text)]])
    print(f"NFC vorkomponiert mit Marke und Hangul LV+T: {len(precomposed)} Proben")

    extra_texts = [
        "\U000200C5\u0301", "\U0002F800\u0301", "\U0001D15E\u0301", "\U0001D1BB\u0301",
        "\U00011099\U000110BA", "\U0001109B\U000110BA", "\U000110A5\U000110BA",
        "\U00011131\U00011127", "\U00011132\U00011127", "\U00011347\U0001133E", "\U00011347\U00011357",
        "\U000114B9\U000114BA", "\U000114B9\U000114B0", "\U000115B8\U000115AF", "\U00011935\U00011930",
        "\U000113C2\U000113C2", "\U0001611E\U0001611E", "\U000113C2\U000113B8",
        "\u1100\u1161", "\u1100\u1161\u11a8", "\uac00\u11a8", "\U000200C5\u0301\u1100\u1161",
        "\u0628\u0650\u08ca", "a\u0316\u1df6", "x\u0301\U000200C5\u0301\u0316",
        "\uf8a1\u0301", "\U0001D160\u0301\u0316",
    ]
    extra = [[[ord(c) for c in t], [ord(c) for c in nfc(t)]] for t in extra_texts]

    layax.write_json(args.out, {
        "source": "tokenizers.normalizers.NFC und pre_tokenizers.Split (Oniguruma)",
        "regex_classes": classes,
        "nfc_single": {str(k): v for k, v in single.items()},
        "nfc_marks": marks,
        "nfc_astral": {str(k): v for k, v in astral.items()},
        "nfc_extra": extra,
        "nfc_precomposed": precomposed,
    })
    print(f"geschrieben: {args.out}")


if __name__ == "__main__":
    main()
