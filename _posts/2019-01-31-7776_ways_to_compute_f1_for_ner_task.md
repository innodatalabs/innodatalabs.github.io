---
title: 7776 ways to compute F1 for a NER task
author: Mike Kroutikov
published: false
---

![image](https://user-images.githubusercontent.com/14280777/52080727-7b7a1100-2566-11e9-8223-37004f620aa7.png)

**TL;DR**: Computing F1 measure on NER task may imply fixing illegal label transitions. There are many ad-hoc ways to do this "fixing". Results can vary widely. Do not do ad-hoc, and use Viterbi decoding on logits.

## Entity encoding schemes

1. IOB - label encoding scheme used in the CoNLL2003 chanllenge. `B-xxx` is only used to split `xxx` entities that are located 
   next to each other.
2. BIO - here `B-xxx` is used at the start of every entity. This encoding seem to work better than IOB.
3. BIOES, BILOU - two names for the same encoding (`U <-> S`, `L <-> E`). Single-token entities are marked with `S-xxx`. Multi-token entities are started with `B-xxx`, ends with `E-xxx`, and middle ones filled with `I-xxx`. This scheme (unlike BIO) is symmetric wrt reversal of sequence order. This makes it attractive for BidiLSTM and CNNs - the neural architectures that do not have preferred "direction".
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

In most NER systems, after NN does its magic we end up having a logits array of shape `[S, C]`, where `S` is the length of the sequence, and `C` is the number of possible labels.

Here is the typical example of computing predictions:
```
logits = ...  # [S, C]
pred = torch.argmax(logits, dim=1)  # [S]
```
This makes a "spot" prediction, picking label without any regard for the neighboring labels. This opens up
the possibility of generating invalid label sequences. Hence we need to "fix" them before decoding entities.

There are many, many ways to do the fixing. If we just look at two labels there are `6^5=7776` of ways to replace invalid label pair with a valid one (for BIOES scheme)!

What if we are not looking at just two labels, but consider wider context? Well, we will have even more ways to fix! And what if we consider the whole sequence? Well, we really should stop right here and get back to the basics.

We want to find label sequence that:
- obeys transition constraints, and
- maximizes the sum of logits for the labels along this path

This task is very well known and can be solved in `O(S*C*C)` using classical Viterbi algorithm. Being quadratic in 
the number of labels, Viterbi can be quite slow. Yet this is the only mathematically optimal way to find entities
from logits.

A  simpler (and faster to compute) "fixing" heuristic is often used instead of Viterbi decoding, and `F1` values are 
reported. But maybe it does not matter and heuristics gives the result that is as good as Viterbi? Lets check!

## Does the way of "fixing" matter?

Here is a standard (easy) NER task: [CoNLL2003 challenge](http://aclweb.org/anthology/W03-0419). And here are 
the SOTA results that cite `F1` scores as the comparison metric: [NLP progress](https://nlpprogress.com/english/named_entity_recognition.html).

I trained a simple GloVe+BidiLSTM model on English NER task, using training parameters from [Jie Yang et al](https://arxiv.org/pdf/1806.04470.pdf) and BIOES label encoding scheme. Then I computed logits for all test samples.

Naive computation of labels (`argmax`) gave me `99` invalid label pairs. This is **1.95%** of the total number of 
"golden" entities in the test set. Hmm, looks like it *may* matter. Changing a label pair can affect up to three entities.

Now, lets try some heuristics for "fixing" bad transitions.

Attempt 1 (close to what is used in [Jie Yang et al](https://arxiv.org/pdf/1806.04470.pdf)):
```python
Entity = collections.namedtuple('Entity', ['label', 'start', 'end'])

def decode_entities_jie(labels):
    pending = None

    for i,l in enumerate(labels):
        if l[:2] == 'B-':
            if pending:
                yield Entity(pending.label, pending.start, i)
                pending = None
            pending = SimpleNamespace(start=i, label=l[2:])
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
Here, input is a list of predicted labels (possibly containing illegal transitions). This function generates triplets
of `label`, `start`, `end`. Where `label` is the entity type (`ORG`, `LOC`, `PER` or `MISC` in this task), and (`start, end`) is a span of tokens for this entity.

Note that when there are no illegal transitions, it does compute correct set of entities. We can ally this to "golden"
labels, then apply to predicted labels and compute `F1` score.

Result is `F1=88.53`. Hmm, sounds pretty good, and right in the interval reported by [Jie Yang et al](https://arxiv.org/pdf/1806.04470.pdf) for the same neural architecture (`F1=88.49+-17`).

Now, note that code above effectively ignores `I` and `O` tags. This does not feel right, right? Right?

Lets try to improve on this by considering all labels.

Attempt 2:
```python
def decode_entities(labels):
    pending = None
    for i,l in enumerate(labels):
        if l[:2] == 'B-':
            if pending:
                ...  # decision point B
            else:
                pending = SimpleNamespace(start=i, label=l[2:])
        elif l[:2] == 'I-':
            if pending is None:
                ...  # decision point I1
            elif pending.label != l[2:]:
                ...  # decision point I2
            else:
                pass
        elif l == 'O':
            if pending is not None:
                ...  # decision point O
        elif l[:2] == 'E-':
            if pending is None:
                ...  # decision E1
            elif pending.label != l[2:]:
                ...  # decision E2
            else:
                yield Entity(pending.label, pending.start, i + 1)
                pending = None
        elif l[:2] == 'S-':
            if pending is not None:
                ...  # decision S
            else:
                yield Entity(label=l[2:], start=i, end=i + 1)
        else:
            raise RuntimeError(f'Unrecognized label: {l}')

    if pending:
        ...  # decision F
```
In the sketch above I replaced with `...` all places where we get unexpected transition and need to fix it. There are total
8 places in the code where fixes are needed.

I used my "best judgement" to pick the resolution. Result is: `F1=87.96`. So much for the "best judgement".

Lets forget about ad-hoc fixing and use Viterbi to decode. Result: `F1=89.29`. Wow! Let me stress, that these are the same logits that gave Jie Yang et al only `F1=88.49+-17`.

## Lets try ALL ways to resolve invalid label pairs

After spectacular failure with my "best judgement" I inserted all possible entity resolution decisions at each point,
made concrete decision be governed py "policy" passed to this function, and generated all possible combinations of
policy decisions. Got total of 5760 combinations (this is a bit less than 7776, because some
bad cases are effectively folded together in my code. For example, invalid transitions `I->B` and `B->B`
are decided in the same place).

Now, lets run all these versions on the logits I have. Result:

Best policy: `F1=88.74`

Worst policy: `F1=87.60`

Interesting. Just using a different ad-hoc policy, one can affect the `F1` score by more than 1%.

Lets display the range of `F1` scores for this NER task:

![image](https://user-images.githubusercontent.com/14280777/52069939-c76c8c00-254d-11e9-857b-a5e8526e6a08.png)

## Side Note 1: Top CRF layer does not (always) guarantee good labels
First, note that using Viterbi decoding on logits has nothing to do with CRF. Here, the purpose of Viterbi is just to enforce
the transition constraints (note that there are no transition weights per se).

If I slap a top CRF layer on top of the neral net and train, will it help me to avoid invalid labels?

It depends.

There is CRF and there is CRF. Some define top CRF layer in a way that only valid transitions are considered (e.g. [Constrained CRF of AllenNLP](https://github.com/allenai/allennlp/blob/89729e041f9163988c9fd6f5592258e11956c431/allennlp/modules/conditional_random_field.py#L324)).

Others allow all transitions, relying on training to discourage bad transitions (e.g. [Jie Yang et al](https://arxiv.org/pdf/1806.04470.pdf)).

The latter does NOT guarantee that output sequence will be legal. Thus, latter will require label "fixing" or constrained decoding.

From my "purist" view, AllenNLP approach is cleaner: use constrained CRF at train and test time.

## Comparing F1 with other results in the literature
At least for CoNLL2003 English NER task, comparison of reported F1 scores should be taken with caution. There is no way to say which F1 is better unless both are computed using the same rules.

Here is some roundup of different approaches. This demonstrates the variety.

1. Emma Strubell, Patrick Verga, David Belanger, and Andrew McCallum. 2017. Fast and accurate entity recognition
with iterated dilated convolutions. In Proceedings of the 2017 Conference on Empirical Methods in Natural
Language Processing, pages 2670–2680. 
   
   Uses [forward greedy search from guessed entity start positions](https://github.com/iesl/dilated-cnn-ner/blob/85147c1b2a59e40a1f241d428b15f2461214d06b/src/eval_f1.py#L49)

2. Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016.
Neural architectures for named entity recognition.
In NAACL
   
   Converts BIOES labels back to IOB labels and uses Perl script to [score](https://github.com/glample/tagger/blob/master/evaluation/conlleval)

3. Jie Yang, Shuailong Liang, Yue Zhang. 2018. Design Challenges and Misconceptions in Neural Sequence Labeling

   Ignores I and O labels, uses B, E, S ones to [resolve](https://github.com/jiesutd/NCRFpp/blob/191e29164a90e65686c6711386ec7e131f1c35e0/utils/metric.py#L73). Also, see above.
   
4. AllenNLP
   
   For BILOU encoding scheme uses forward search, raising exception on any invalid label [sequence](https://github.com/allenai/allennlp/blob/4674b0182187ef10c54d0578d97f4ba9769a2863/allennlp/data/dataset_readers/dataset_utils/span_utils.py#L217). Apparently, this encoding scheme is only used with constrained CRF top layer.
   
   For BMES label encoding scheme, uses heuristics to resolve ill-formed [entities](https://github.com/allenai/allennlp/blob/4674b0182187ef10c54d0578d97f4ba9769a2863/allennlp/data/dataset_readers/dataset_utils/span_utils.py#L376)

## Summary

1. Use Viterbi
2. Use Viterbi, pleeease
3. When you see F1 score reported for NER task - check how it was computed. This matters!
4. When somebody uses CRF, check how CRF is defined: is it constrained CRF or not? If not, how labels were fixed?
