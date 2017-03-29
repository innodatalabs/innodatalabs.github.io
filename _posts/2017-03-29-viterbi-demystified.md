---
title: Viterbi algorithm, demystified
author: Mike Kroutikov
math: true
published: true
---
When dealing with sequences, Viterbi algorithm and Viterbi decoding pops up regularly. This algorithm is usually described in the
context of Hidden Markov Models. However, the application of this algorithm is not limited to HMMs. Besides, HMMs lately fell out
of fashion as better Machine Learning techniques have been developed.

## Definitions
Bear with me here, please. This is the most tedious and boring part - explaining the notations.

We have a discrete sequence that starts at index $$t=1$$ and ends at index $$t=T$$. Here $$T$$ is the length of the sequence.
At each index $$t$$ there are $$S$$ possible states. The sequence (sometimes called _path_) is when we decide which state at each
index was chosen. State valiable $$s(t)$$ can take any value in the range $$1\dots S$$ at each index $$t$$.

Thus, we describe sequence as $$s(t)$$, $$t=1\dots T$$. Number of possible sequences (paths) grows exponentially with the length of the 
sequence $$T$$.

One example: we want to describe (or predict) weather for a week. Here the length of our sequence is 7 (number of days), and
each day we have either `rainy`, `cloudy` or `sunny` state. We have total of three states. Let us encode `rainy=0`, `cloudy=1`, 
and `sunny=2`.

Here is one possible sequence:
```
sunny(2)  
sunny(2)  
cloudy(1)  
rainy(0) 
rainy(0) 
cloudy(1) 
sunny(2)
```
Or, in short, our `s(t)` is:
```
2 2 1 0 0 1 2
```

There are total $$3^7=2187$$ possible weather sequences in this model ($$T=7$$, $$S=3$$).

Now, if we want to predict the weather for a week, we need to come up with an algorithm that finds the most likely
sequence $$s(t)$$. In ML this task is cast into optimization of some objective function $$L[s]$$. 
This function depends on the sequence, and we want to find $$s(t)$$ that minimizes the $$L$$. 
When I write $$L[s]$$ I mean that $$L$$ is a real number that depends on 
complete path, i.e. depends on the state choice at every index $$t$$. Same cam be written as $$L[s] = L(s(1), s(2), \dots s(T))$$.

Practically, there are specific popular forms of $$L$$ dependency on $$s$$. Choice of $$L$$ is super important, because it defines
the model. Good choice will predict weather reliably. Bad choice will fail to do so.

### Local loss
Somewhat trivial, but still reasonable model.

$$
L[s] = \sum_{t=1}^T l(t, s(t))
$$

Here we introduced $$l(t, s)$$, which we will call *logit* array (or matrix). 
For each index $$t$$ we have $$S$$ numbers that tell us how likely the
corresponding state is. For example, for weather prediction we may have the following logits:
```
index      rainy=0    cloudy=1    sunny=2
-----------------------------------------
t=1        -0.1       -3.5        2.3
t=2        -0.8       -2.5        1.3
t=3        -1.2       -1.0        4.3
t=4        -0.2       -3.0        0.1
t=5        0.15       0.2        -2.7
t=6        0.19       1.5        -2.8
t=7        0.7        3.5        -5.3
-----------------------------------------
```
Where do these logits come from? Typically from the top Neural Network layer. But for the discussion of Viterbi this is not important.
What is important is that our model $$L[s]$$ is fully described by a $$3\times 7$$ matrix of logits. 
Once we have these logits computed, we
can find optimal sequence $$s^*(t)$$ by minimizing the $$L$$.

Even though we have 2187 possible paths to consider, finding best sequence can be done by just picking the smallest logit at each
index. Thus, we can efficiently compute the best sequence $$s^* = [1, 1, 0, 1, 2, 2, 2]$$ and predict this:
```
cloudy
cloudy
rainy
cloudy
sunny
sunny
sunny
```

### Pairwise loss
In the previous model, solution was computed locally, for each index $$t$$. That was possible because optimization objective $$L$$ was
a sum of local objectives.

For the weather prediction problem one can notice that in practice weather does not switch from `rainy` to `sunny` or from `sunny` 
to `rainy`. It (almost) always goes via `cloudy`. To capture this observation, we can introduce a better loss:

$$
L[s] = \sum_{t=1}^T l(t, s(t)) + \sum_{t=1}^{T-1} m(t, s(t), s(t+1))
$$

Here first term is the same as before - the sum of local losses. The second term allows us to model dependencies between
today and tomorrow. We introduced a matrix $$m(t, s, q)$$ that controls the transitions. A physisist will say that we have
"a potential with pairwise interactions". In the context of HMMs term $$m$$ is called *transition probabilities*.

For a given $$t$$, $$m$$ is a matrix $$S\times S$$ that tells us how probable is the transition from state $$s$$ to state $$q$$.
In our weather prediction example, such a matrix may look like:
```
   0.0    2.3    1000.0
   5.3    1.5       4.2
1000.0    3.3       0.1
```
If we choose a path that switches from `rainy(0)` to `cloudy(1)` we will get additional loss of $$m(t, 0, 1)$$, or 5.3.
Note that for simplicity in this example $$m$$ is the same for all indices $$t$$.

Now, if we choose a path that switches from `rainy(0)` to `sunny(2)` we will be penalized by extra loss of 1000.

## Viterbi algoritm
Brute-force way of finding optimal sequence in a pairwise loss model is prohibitively expensive. Just think about a problem
with $$S=10$$ and sequence length $$T=80$$. 
The number of all possible sequences in this problem is $$10^{80}$$, which is 
[how many atoms we have in the observable universe](http://www.universetoday.com/36302/atoms-in-the-universe/). 
Good luck computing this!

Yet we can efficiently find the best sequence exploiting the pairwise structure of the model.

### Dynamic programming to the resque
Viterbi algorithm is an instance of a dynamic programming class of algorithms.

First that we do in DP is *we pretend we know the solution*. To elaborate, lets first introduce a constrained optimization problem:
we asked to optimize objective function $$L[s]$$ with the additional condition that sequence $$s(t)$$ ends at a predefined state $$q$$.
I will write this optimization objective as:

$$
L[s \vert s(T)=q]
$$

where $$q$$ can take any value in the range $$1\dots S$$.

If we know solution for the constrained problem, we can find the full solution by just cycling thru all values of $$q$$ and finding the
one that yields the minimal value of $$L$$. Thus, knowing $$L[s \vert s(T)=q]$$ we can easily give the answer to the original problem.

Now, back to the Dynamic Programming approach. We pretend that *we know the answer already for a bit shorter problem*. 
Specifically, we pretend that we can easily compute the best constrained objective $$L[s \vert s(T-1)=q]$$. Assuming this,
we notice that to find the solution to the original problem we just need to consider the last step of our sequence.

Lets express out pairwise objective function thru the objective of a one step smaller problem:

$$
L_T = \sum_{t=1}^T l(t, s(t)) + \sum_{t=1}^{T-1} m(t, s(t), s(t+1))
$$

$$
L_T = \sum_{t=1}^{T-1} l(t, s(t)) + l(T, s(T)) + \sum_{t=1}^{T-2} m(t, s(t), s(t+1)) + m(T-1, s(T-1), s(T))
$$

$$
L_T = L_{T-1} + l(T, s(T)) + m(T-1, s(T-1), s(T))
$$

Now, if we know solution $$L^*_{T-1}(q)$$ to a constrained problem of size $$T-1$$, we can find a solution to the
constrained problem of size $$T$$, $$L^*_T(r)$$, because:

$$
L^*_T(r) = \textrm{argmin}_q L^*_{T-1} + l(T, r) + m(T-1, q, r)
$$

To finsh the hard part, lets note that solution for a problem with size 1 is super easy (as there is no pairwise term
in the optimization objective). Viterbi will start with sequsence of size 1 and grow the solution till it reaches final
length of $$T$$. The complexity is $$O(S^2T)$$, and space needed is $$O(ST)$$.

Again, returning to our imaginary problem that has 10 states and sequence of length 80, computational complexity is
$$O(10^2 80) = O(8000)$$, and space needed is $$O(800)$$ - very manageable!

## Viterbi and transition constraints
Traditionally, Viterbi is used in ML methods like [HMMs](https://en.wikipedia.org/wiki/Hidden_Markov_model)
and [CRFs](https://en.wikipedia.org/wiki/Conditional_random_field) where transition matrix is learned as part of the
model training.

In the classical sequence labeling (e.g. [POS tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)) only logits
are used and local loss is used for label decoding.

However, we can still use Viterbi if we add an ad-hoc transition matrix that expresses some additional knowledge.

For example, if we want to enforce a rule that a sunny day is never followed by a rainy day, we just postulate
that transition matrix $$m$$ is the same for all $$t$$ and has the following structure:
```
   0     0     0
   0     0     0
1000     0     0
```
Then, we can use the same logits to decode the sequence of weather predictions. It is guaranteed to obey this
constraint!

If additionally we want to forbid transitions from rainy to sunny, we would use the following transition matrix:
```
    0    0 -1000
    0    0     0
-1000    0     0
```

Thus, we are taking the original solution to a sequence labeling problem, and add external constraint. Then we find the best
sequence that minimizes loss under the given constraint.

To recap, we can use Viterbi to find solutions to sequence labeling problems under additional transition constrains.

## IOB constraint
One important transition constraint arises when one uses popular
[IOB label encoding](https://en.wikipedia.org/wiki/Inside_Outside_Beginning).

Indeed, in IOB sequence there are some transitions that are not expected, given the meaning of IOB labeling. For example,
well-formed IOB sequence uses `B` label to mark the start of an interval:
```
in    O
South B-location
Korea I-location
and   O
China B-location
```

But this one does not make sense:
```
in    O
South I-location
Korea I-location
and   O
China B-location
```

Formally, the rules are: label 'I' can only be preceded by 'I' or 'B'. In other words, transition from 'O' to 'I' is not allowed.

This can readily be expressed in terms of a transition matrix, and Viterbi is used to ensure that we always return a sensible 
IOB label sequence.

## XML structure constraint
When doing prediction on a text of a structural document (think XML), there are additional constraints if we want to
express our predictions as additional XML tags. The additional XML tags we are adding should not contradict the
hierarchical structure of XML document.

Again, these constrains can be expressed in terms of transition matrix. This time for every position in our sequence we will have
a different constraint - we can no longer use $$m$$ that does not depend on index $$t$$. So building the matrix $$m$$ becomes more
complicated. But onse it is built, we will employ standard Viterbi decoder and get back label sequence that is guaranteed to
play well with the structure of the input XML document.

## Viterbi extensions
Apart from asking Viterbi algorithm to find the best sequence, one can ask: give me the *next best* sequence. A greedy client can
even demand this: give me $$N$$ best sequences, sorted by the "bestness".

Luckily, a simple modification to Viterbi algorithm allows one to efficiently compute *next best* and *next next best*, and so on.

This can be used to estimate the confidence and suspicious places in the predicted sequence. The informal reasoning is this:
lets get 10 best decodings and compare the very best decoding to the rest 9 decodings.

If there is a significant drop in the value of loss function when we go from the very best sequence to the next best, then
system is quite confident in the prediction. Conversely, if losses of the very best and next best decoding are similar, then
system thinks that both sequences are equally likely. Look where they differ and that would be a location where machine
is not sure with the prediction.

## Summary
When doing sequence labeling, Viterbi algorithm is very useful and broadly applicable - get to know it!
