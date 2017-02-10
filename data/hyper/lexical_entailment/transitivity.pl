isa(X, Z) :- isa(X,Y), isa(Y,Z)
!isa(X,Y) :- isa(Y,X)