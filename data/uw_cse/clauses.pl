advisedBy(Y,X):- publication(P,X), publication(P,Y), unary(Y,student)
unary(student, S):- advisedBy(S,P)
unary(professor, P):- advisedBy(S,P)