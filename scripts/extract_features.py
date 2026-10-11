"""Build the feature table for the grammar-scoring model.

Reads data/transcripts.csv (whisper output: text + timing stats) and
computes, per file:
  - prosodic features (speech rate, pause ratios) from whisper timings
  - syntactic/grammar features from spaCy (POS ratios, fragments, parse depth,
    disfluency markers, type-token ratio)
  - a fluency feature: GPT-2 pseudo-perplexity of the transcript (how
    "surprising" the word sequence is to a language model -- low-grammar
    speech tends to score as less predictable / higher perplexity)

Writes data/features.csv, one row per file, joinable with train.csv/test.csv
on `filename`.
"""
import math
import os

import pandas as pd
import spacy
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTS = os.path.join(ROOT, "data", "transcripts.csv")
OUT = os.path.join(ROOT, "data", "features.csv")

FILLERS = {"um", "uh", "uhh", "umm", "hmm", "er", "ah", "like", "you know", "i mean"}


def load_gpt2():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tok, model


@torch.no_grad()
def gpt2_perplexity(text, tok, model):
    if not text.strip():
        return 0.0
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return 0.0
    out = model(input_ids, labels=input_ids)
    return math.exp(min(out.loss.item(), 20))  # cap to avoid inf on garbage


def disfluency_rate(text_lower, n_words):
    if n_words == 0:
        return 0.0
    count = 0
    for filler in FILLERS:
        count += text_lower.count(filler)
    return count / n_words


def repeated_bigram_rate(tokens):
    if len(tokens) < 2:
        return 0.0
    reps = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    return reps / len(tokens)


def spacy_features(doc, n_words_whisper):
    sents = list(doc.sents)
    n_sents = max(len(sents), 1)
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    n_tok = max(len(tokens), 1)

    pos_counts = {}
    for t in tokens:
        pos_counts[t.pos_] = pos_counts.get(t.pos_, 0) + 1
    pos_ratio = {f"pos_{p}": c / n_tok for p, c in pos_counts.items()}

    sent_lens = [len([t for t in s if not t.is_space and not t.is_punct]) for s in sents]
    avg_sent_len = sum(sent_lens) / n_sents
    frag_sents = sum(1 for l in sent_lens if l <= 2)
    runon_sents = sum(1 for l in sent_lens if l >= 35)

    # sentences with no verb / no root verb -> likely ungrammatical fragments
    no_verb_sents = 0
    max_tree_depth = 0
    for s in sents:
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in s)
        if not has_verb:
            no_verb_sents += 1
        for t in s:
            depth = 0
            cur = t
            while cur.head != cur:
                depth += 1
                cur = cur.head
                if depth > 50:
                    break
            max_tree_depth = max(max_tree_depth, depth)

    lemmas = [t.lemma_.lower() for t in tokens]
    ttr = len(set(lemmas)) / n_tok

    feats = {
        "n_sentences": n_sents,
        "avg_sentence_len": avg_sent_len,
        "frag_sentence_ratio": frag_sents / n_sents,
        "runon_sentence_ratio": runon_sents / n_sents,
        "no_verb_sentence_ratio": no_verb_sents / n_sents,
        "max_parse_depth": max_tree_depth,
        "type_token_ratio": ttr,
    }
    feats.update(pos_ratio)
    return feats


def main():
    df = pd.read_csv(TRANSCRIPTS)
    df["text"] = df["text"].fillna("")

    print("Loading spaCy model...", flush=True)
    nlp = spacy.load("en_core_web_sm")
    print("Loading GPT-2...", flush=True)
    tok, gpt2 = load_gpt2()

    rows = []
    for i, r in df.iterrows():
        text = r["text"].strip()
        text_lower = text.lower()
        n_words = r["n_words"] if r["n_words"] else len(text.split())
        duration = max(r["duration_sec"], 0.1)

        speech_rate = n_words / duration
        speaking_time = max(duration - r["total_pause_sec"], 0.1)
        articulation_rate = n_words / speaking_time
        pause_ratio = min(r["total_pause_sec"] / duration, 1.0)

        doc = nlp(text) if text else nlp("")
        sfeats = spacy_features(doc, n_words)

        tokens_lower = text_lower.split()
        dis_rate = disfluency_rate(text_lower, n_words)
        rep_rate = repeated_bigram_rate(tokens_lower)
        ppl = gpt2_perplexity(text, tok, gpt2)

        row = {
            "split": r["split"],
            "filename": r["filename"],
            "duration_sec": duration,
            "n_words": n_words,
            "speech_rate_wps": speech_rate,
            "articulation_rate_wps": articulation_rate,
            "pause_ratio": pause_ratio,
            "max_pause_sec": r["max_pause_sec"],
            "avg_word_conf": r["avg_word_conf"],
            "disfluency_rate": dis_rate,
            "repeated_bigram_rate": rep_rate,
            "gpt2_perplexity": ppl,
            "gpt2_log_perplexity": math.log(max(ppl, 1.0)),
        }
        row.update(sfeats)
        rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(df)}] features done", flush=True)

    out_df = pd.DataFrame(rows).fillna(0.0)
    out_df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(out_df)} rows, {out_df.shape[1]} columns")


if __name__ == "__main__":
    main()
