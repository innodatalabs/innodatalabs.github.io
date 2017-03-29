---
title: Viterbi algorithm, demystified
author: Mike Kroutikov
math: true
published: false
---
When dealing with sequences, Viterbi algorithm and Viterbi decoding pops up regularly. This algorithm is usually described in the
context of Hidden Markov Models. However, the application of this algorithm is not limited to HMMs. Besides, HMMs lately fell out
of fashion as better Machine Learning techniques have been developed.

## Definitions
Bear with me here, please. This is the most tedious and boring part - to explain the notations.

We have a discrete sequence that starts at index `t=1` and ends at index `t=T`. Here `T` is the length of the sequence.
At each index `t` there are `S` possible states. The sequence (sometimes called _path_) is when we decide which state at each
index was chosen. State valiable `s(t)` can take any value in the range `1..S` at each index `t`.

Thus, we describe sequence as `s(t), t=1..T`. Number of possible sequences (paths) grows exponentially with the length of the 
sequence `T`.

One example: we want to describe (or predict) weather for a week. Here the length of our sequence is 7 (number of days), and
each day we have either `rainy`, `cloudy` or `sunny` state. We have total of three states. Let us encode `rainy=0`, `cloudy=1`, 
and `sunny=2`.

Here is one possible sequence:
```
sunny(2)  sunny(2)  cloudy(1)  rainy(0) rainy(0) cloudy(1) sunny(2)
```
Or, in short, our `s(t)` is:
```
2 2 1 0 0 1 2
```

There are total `3^7=2187` possible weather sequences in this model. `T=7`, `S=3`.

Now, if we want to predict the weather for a week, we need to come up with an algorithm that finds the most likely
sequence `s(t)`. In ML this task is cast into optimization of some objective function `L[s]`. This function depends on the
sequence, and we want to find `s(t)` that minimized the `L`. When I write `L[s]` I mean that `L` is a real number that depends on 
complete path, i.e. depends on the state choice at every index `t`. Same cam be written as `L[s] = L(s(1), s(2), .. s(T))`.

Practically, there are specific popular forms of `L` dependency on `s`. Choice of `L` is very important, because it defines
the model. Good choice will predict weather reliable. Bad choice will fail.

### Local loss
Somewhat trivial, but still.
$$
L[s] = \sum_{t=1}^T l(t, s(t))
$$
Here we introduced `l(t, s)`, which we will call *logit* array (or matrix). 
For each index we have `S` numbers that tell us how likely the
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
What is important is that our model `L[s]` is fully described by a 3x7 matrix of logits. Once we have these logits computed, we
can find optimal sequence `s*(t)` by minimizing the `L`.

Even though we have 2187 possible paths to consider, finding best sequence can be done by just picking the smallest logit at each
index. Thus, we can efficiently compute the best sequence `s* = [1, 1, 0, 1, 2, 2, 2]` and predict
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
In the previous model, solution was computed locally, for each index `t`. That was possible because optimization objective `L` was
a sum of local objectives.

For the weather prediction problem one can notice that in practice weather does not switch from `rainy` to `sunny` or from `sunny` 
to `rainy`. It (almost) always goes via `cloudy`. To capture this observation, we can introduce a better loss:
```
L[s] = \sum_{t=1}^T l(t, s(t)) + \sum_{t=1}^{T-1} m(t, s(t), s(t+1))
```

Here first term is the same as before - the sum of local losses. The second term allows us to model dependencies between
today and tomorrow. We introduced a matrix `m(t, s, q)` that controls the transitions. A physisist will say that we have
potential with pairwise interactions. In the context of HMMs term `m` is called *transition probabilities*.

For a given `t`, `m` is a matrix `SxS` that tells us how probable is the transition from state `s` to state `q`.
In our weather predition example, such a matrix may look like:
```
0       2.3     1000.0
5.3     1.5    4.2
1000.0  3.3    0.11
```
If we choose a path that switches from `rainy(0)` to `cloudy(1)` we will get additional loss of `m(t, 0, 1)`, or 5.3.
Note that for simplicity in this example `m` is the same for all indices `t`.
Now, if we choose a path that switches from `rainy(0)` to `sunny(2)` we will be penalized by extra loss of 1000.

## Viterbi algoritm
Brute-force way of finding optimal sequence in a pairwise loss model is prohibitively expensive. Just think about a problem
with `S=10` and sequence length `T=80`. The number of all possible sequences in this problem is `10^80`, which is [how many atoms
we have in the observable universe](http://www.universetoday.com/36302/atoms-in-the-universe/). Good luck computing this!

Yet we can efficiently find the best sequence exploiting the pairwise structure of the model.
