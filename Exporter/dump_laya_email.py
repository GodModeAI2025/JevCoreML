#!/usr/bin/env python3
"""Referenz für laya.email: bereinigte Mailtexte und Zustände, aus laya selbst.

Zwei Quellen. Handverlesene Fälle für die Stellen, an denen Python-Regex und ICU verschieden
denken: Leerraum jenseits von ASCII, Zeilentrenner, `\u017f` und das Kelvinzeichen unter re.I, die
Signaturgrenze bei 60 % der Zeilen. Dazu erzeugte Mails aus Zeilenbausteinen, mit festem Seed,
damit der Korpus reproduzierbar bleibt und trotzdem Kombinationen trifft, an die niemand denkt.
"""
import argparse
import random

import layax
from laya.email import clean_email_body, email_state

HANDPICKED = [
    "",
    "   ",
    "Hallo,\n\nbitte die Rechnung korrigieren.\n\nDanke",
    "Hi team,\nthe export fails since Monday.\n\nOn Mon, 3 Mar 2025 at 10:00, Bob <bob@x.com> wrote:\n> old text\n> more",
    "On Mon, Bob wrote:\nthis header is first, so it stays\nand the text continues",
    "Short body\r\nwith CRLF\r\nlines\rand a lone CR",
    "Literal backslash n\\nin the text\\n\\nand again",
    "Text\n-----Original Message-----\nFrom: someone\nold",
    "Text\n----- Forwarded message -----\nstuff",
    "Text\n________________\nquoted",
    "Text\nFrom: Alice <a@x.com>\nSent: yesterday",
    "> quoted first\n>> nested\nreal text\n   > indented quote",
    "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10\nThanks,\nBob",
    "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nBest regards,\nAnna\nACME Corp",
    "Please help.\n--\nSignature line\nmore signature",
    "Please help.\n\nSent from my iPhone",
    "One\nTwo\nThree\nKind regards and a very long closing sentence that is well over forty chars",
    "Body text here.\n\nThis email is confidential and intended solely for the use of the addressee.\n\nMore body.",
    "Body.\n\nIf you have received this e-mail in error, please delete it.",
    "Body.\n\nIf you received this message in error notify us.",
    "tabs\tand   spaces\t\tcollapse   here",
    "unicode\u00a0nbsp\u2003em space\u3000ideographic",
    "line with trailing nbsp\u00a0\u00a0\nnext",
    "sep\u2028inside a line\u2029and paragraph sep\x85next line char",
    "info separators\x1c\x1d\x1e\x1fend",
    "X\n\u00a0\n\u00a0\nY",
    "A\n\n\n\nB",
    "A\n \t \nB",
    "\u017fincerely text\nline\nline\nline\nline\u017fincerely",
    "p\nq\nr\ns\nt\nu\nv\nw\n\u212aind regards",
    "p\nq\nr\ns\nt\nu\nv\nw\nTHANK YOU!!!",
    "p\nq\nr\ns\nt\nu\nv\nw\ncheers_mate 2025",
    "p\nq\nr\ns\nt\nu\nv\nw\ncheers\u0301",
    "Emoji \U0001F600 and combining e\u0301 text",
    "x" * 3500,
    ("\U0001F600" * 2999) + "abc",
    "On " + "x" * 299 + " wrote:",
    "Text\nOn " + "x" * 300 + " wrote:\nquoted",
    "Text\nOn " + "x" * 301 + " wrote:\nquoted",
    "Text\n  On Tuesday wrote:  \nquoted",
    "Text\nON TUESDAY WROTE:\nquoted",
    "Text\nFrom:\tsomeone\nquoted",
    "Text\nFrom:someone\nnot a header",
]

CONTENT = [
    "Hi team,", "Hello,", "Guten Tag,", "The invoice lists the same item twice.",
    "Can you fix it before Friday?", "Our deployment fails with error 503.",
    "Bitte um Rückruf.", "Merci pour votre aide.", "We need access for three users.",
    "   leading spaces here", "trailing spaces   ", "\ttab start", "x", "",
    "Order #4411 was charged twice.", "Please advise.", "cheers for the update, the build is green",
]
QUOTE = [
    "On Mon, 3 Mar 2025, Bob <bob@example.com> wrote:", "-----Original Message-----",
    "---- Forwarded Message ----", "________", "From: Alice <alice@example.com>",
    "> quoted line", ">> nested quote", "   > indented quote",
]
SIGNATURE = [
    "--", "-- ", "Best,", "Best regards,", "Kind regards", "Thanks!", "Thank you.",
    "Many thanks, Bob", "Regards", "Cheers", "Sincerely,", "Warm wishes",
    "Sent from my iPhone", "Sent from my Android device", "thanks a lot for everything you did this week",
]
DISCLAIMER = [
    "This message is confidential.",
    "It is intended solely for the use of the named addressee.",
    "If you have received this email in error, please notify the sender.",
    "Intended for the recipient only.",
]
SPACES = ["", " ", "  ", "\t", "\u00a0", "\u2003", "\u3000"]


def generated(count, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        lines = []
        for _ in range(rng.randint(0, 24)):
            kind = rng.random()
            if kind < 0.55:
                line = rng.choice(CONTENT)
            elif kind < 0.68:
                line = rng.choice(QUOTE)
            elif kind < 0.86:
                line = rng.choice(SIGNATURE)
            elif kind < 0.94:
                line = rng.choice(DISCLAIMER)
            else:
                line = ""
            if rng.random() < 0.15:
                line = rng.choice(SPACES) + line + rng.choice(SPACES)
            lines.append(line)
        sep = rng.choice(["\n", "\n", "\n", "\r\n", "\r", "\\n"])
        out.append(sep.join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4000)
    ap.add_argument("--out", default=str(layax.ROOT / "Golden" / "laya" / "email.json"))
    args = ap.parse_args()

    bodies = HANDPICKED + generated(args.count, seed=20260921)
    cases = [{"body": b, "clean": clean_email_body(b), "clean_200": clean_email_body(b, max_chars=200)}
             for b in bodies]
    states = []
    for subject, body, sender, clean, extra in [
        ("  Invoice #4411  ", "Hi,\nplease fix.\n\nThanks,\nBob", "bob@x.com", True, {}),
        (None, None, None, True, {}),
        ("Re: order", "raw\n> quote kept because clean=False", "", False, {}),
        # "from" überschreibt den Absender an seiner Stelle. "subject" oder "body" über extra
        # lässt Python gar nicht zu, die kollidieren mit den Parametern.
        ("S", "B", "s@x.com", True, {"priority": "high", "note": None, "from": "override@x.com"}),
        ("S", "B", None, True, {"thread_id": 42, "labels": ["a", "b"]}),
    ]:
        state = email_state(subject, body, sender=sender, clean=clean, **extra)
        states.append({"subject": subject, "body": body, "sender": sender, "clean": clean,
                       "extra": [[k, v] for k, v in extra.items()], "state": state})
    layax.write_json(args.out, {"handpicked": len(HANDPICKED), "count": len(cases),
                                "cases": cases, "states": states})
    print(f"{len(cases)} Mails ({len(HANDPICKED)} handverlesen), {len(states)} Zustände -> {args.out}")


if __name__ == "__main__":
    main()
