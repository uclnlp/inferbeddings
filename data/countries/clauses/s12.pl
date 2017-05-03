locatedIn(C, R) :- locatedIn(C, S), locatedIn(S, R)
locatedIn(C1, R) :- neighborOf(C1, C2), locatedIn(C2, R)
