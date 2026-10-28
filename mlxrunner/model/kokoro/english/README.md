# Native Kokoro English frontend

English only. Dictionary-first pronunciations, number/currency/abbreviation
expansion, small context rules for heteronyms, and a native letter-to-sound
fallback for unfamiliar Latin words. This does not reproduce spaCy POS tagging;
unknown-word pronunciation and ambiguous context remain approximate. Unsupported
symbols return an error instead of silently deleting text.

US and British gold/silver dictionaries are unmodified JSON payloads from the
owner-authorized Misaki 0.9.4 package, gzip packaged for embedding. The upstream
project is https://github.com/hexgrad/misaki and publishes Apache-2.0 licensing.
`data/LICENSE` retains that license; `data/provenance.json` pins exact payload and
compressed hashes. No Python, spaCy model, executable or runtime download is used.

The frontend implementation is new Go code. It maps the dictionary's American
flap to Kokoro's `T` convention and does not claim complete Misaki equivalence.
