---
title: 7776 ways to compute F1 for a NER task
author: Mike Kroutikov
published: false
---

TL;DR: Computing F1 measure on NER task implies fixing illegal label transitions. There are many ad-hoc ways to do this "fixing". Result can vary widely. Do not do ad-hoc, and use Viterbi decoding for the best results.

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
Ben/I-PER was/O in/O New/I-LOC York/I-LOC City/I-LOC on/O Monday/O
```
Here I used suffix notation to show label of each word. For clarity, I will omit `/O` from now on:

IOB:
```
Ben/I-PER was in New/I-LOC York/I-LOC City/I-LOC on Monday
```

BIO:
```
Ben/B-PER was in New/B-LOC York/I-LOC City/I-LOC on Monday
```

BIOES:
```
Ben/S-PER was in New/B-LOC York/I-LOC City/E-LOC on Monday
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

Naive computation of labels gave me 99 invalid label pairs. This is 1.95% of the total number of "golden" entities in the test set. Hmm, looks like it may matter.

Using Viterbi decoding, resulted in `F1=89.29`.

Now, lets try some heuristics.

Attempt 1 (close to what is used in [??](??}):
```python
Entity = collections.namedtuple('Entity', ['label', 'start', 'end'])

def decode_entities_jie_bioes(labels):
    pending = None

    for i,l in enumerate(labels):
        if l[:2] == 'B-':
            if pending:
                yield Entity(pending.label, pending.start, i)
                pending = None
            pending = SimpleNamespace(start=i, end=i+1, label=l[2:])
        elif l[:2] == 'S-':
            if pending is not None:
                yield Entity(pending.label, pending.start, i)
                pending = None
            yield Entity(label=l[2:], start=i, end=i+1)
        elif l[:2] == 'E-':
            if pending is not None:  # jie does not check if B- uses the same label!
                yield Entity(pending.label, pending.start, i + 1)
                pending = None

    if pending:
        yield Entity(pending.label, pending.start, pending.end)
```
Result is `F1=88.53`. Hmm, very different from Viterbi.

Note that this code effectively ignores `I` and `O` tags.

Lets try to improve on this, by considering all labels.

Attempt 2:
```
def decode_entities(labels):
    pending = None
    for i,l in enumerate(labels):
        if l[:2] == 'B-':
            if pending:
                ...
            else:
                pending = SimpleNamespace(start=i, end=i+1, label=l[2:])
        elif l[:2] == 'I-':
            if pending is None:
                ...
            elif pending.label != l[2:]:
                ...
            else:
                pending.end += 1
        elif l == 'O':
            if pending is not None:
                ...
        elif l[:2] == 'E-':
            if pending is None:
                ...
            elif pending.label != l[2:]:
                ...
            else:
                pending.end += 1
                yield Entity(pending.label, pending.start, pending.end)
                pending = None
        elif l[:2] == 'S-':
            if pending is not None:
                ...
            else:
                yield Entity(label=l[2:], start=i, end=i+1)
        else:
            raise RuntimeError(f'Unrecognized label: {l}')

    if pending:
        ...
```
In the sketch above I replaced with `...` all places where we get unexpected transition and need to fix it.

I used my "best judgement" to pick the resolution. Result is: `F1=87.96`. So much for the "best judgement".

## Gauging all ways to resolve invalid label pairs

After spectacular failure with my "best judgement" I inserted all possible entity resolution decisions at each point,
made concrete decision be governed py "policy" passed to this function, and generated all possible combinations of
policy decisions. Got total of 2880 combinations (this is less than maximum possible value of 7776, because some
bad cases are effectively folded together in the code above).

Best policy: `F1=88.74`

Worst policy: `F1=87.60`

## Summary

![image](https://user-images.githubusercontent.com/14280777/52069939-c76c8c00-254d-11e9-857b-a5e8526e6a08.png)

1. Use Viterbi
2. Use Viterbi
3. Use Viterbi
