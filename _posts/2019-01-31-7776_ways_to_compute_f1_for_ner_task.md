---
title: 7776 ways to compute F1 for a NER task
author: Mike Kroutikov
published: false
---

TL;DR: Computing F1 measure on NER task implies fixing illegal label transitions. There are many ad-hoc ways to do this "fixing". Result can vary widely.

## Entity encoding schemes

1. IOB - the original one used in the CoNLL2003 dataset. `B-xxx` is only used to split `xxx` entities that are located 
   next to each other.
2. BIO - here `B-xxx` is used at the start of every entity. This encoding seem to work better that IOB.
3. BIOES, BILOU - two names for the same encoding (`U <-> S`, `L <-> E`). Single-token entities are marked with `S-xxx`. Multi-token entities are started with `B-xxx`, ends with `E-xxx`, and middle ones filled with `I-xxx`.
4. BMES - no clue what this is

Example:
```
Ben was in New Your City on Monday
```

IOB:
```
Ben##I-PER was##O in##O New##I-LOC York##I-LOC City##I-LOC on##O Monday##O
```
Here I used suffix notation to show label of each word. For clarity, I will omit `##O` from now on:

IOB:
```
Ben##I-PER was in New##I-LOC York##I-LOC City##I-LOC on Monday
```

BIO:
```
Ben##B-PER was in New##B-LOC York##I-LOC City##I-LOC on Monday
```

BIOES:
```
Ben##S-PER was in New##B-LOC York##I-LOC City##E-LOC on Monday
```

## Transition rules for BIOES

Each scheme has its own transition rules. Lets look closer at BIOES encoding rules:

* `B-xxx` : starts entity `xxx`, allowed after `O`, `S-yyy`, `E-yyy` (with `yyy` being any entity), or as a first one in the sequence
* `I-xxx` : continues entity `xxxx`, allowed after `I-xxx` or `B-xxx` only
* `E-xxx` : ends entity `xxx`, allowed after `I-xxx` or `B-xxx` only
* `S-xxx` - starts and ends a single-token entity `xxx`, allowed after `O`, `S-yyy`, `E-yyy` (with `yyy` being any entity), or as a first one in the sequence
* `O` - no entity, allowed after `O`, `S-yyy`, `E-yyy` (with `yyy` being any label), or as the fist one in a sequence

## How to decode entities from logits

In most NER systems, after NN does its magic we end up having a logits array of shape `[S, C]`, where S is the length of the sequence, and
C is the number of possible labels.

From this point we have a choice:
* to do it "Right" and use Viterbi decoding, finding the best possible label path that (a) obeys transition constraints, and (b) maximises the sum of logits along this path.
* do it fast and dirty way: compute best local labels (ignoring transition constraints), and then fix the bad transitions.

Typically, people will do the latter. Reasons being:
* Viterbi is slow
* Viterbi is hard to get right
* Just do not know better

Here is the typical example of computing predictions:
```
logits = ...  # [S, C]
pred = torch.argmax(logits, dim=1)  # [S]
```
This makes a "spot" prediction, picking label at time `t` without any regard for the neighboring labels.

There are many ways to "fix" the predictions. In case of BMES we have `6^5 = 7776` ways of fixing invalid transitions, if
we are only looking at two labels at a time - current and previous.

## Does the way of "fixing" matter?

Here is a standard (easy) NER task: [CoNLL2003 challenge](http://aclweb.org/anthology/W03-0419). And here are the SOTA results
that cite F1 scores as the comparison metric: [NLP progress](https://nlpprogress.com/english/named_entity_recognition.html).

I trained a simple GloVe+BidiLSTM model, using training parameters from (???)[???], and saved all the logits for the test set.

Naive computation of labels gave me 99 invalid label pairs. This is 1.95% of the total number of "golden" entities in the test set.

Using Viterbi decoding, resulted in `F1=89.29`.

Now, lets try some heuristics.
