### Jointly Embedding Knowledge Graphs and Logical Rules

```
$ mkdir data ; cd data
$ wget -c http://www.aclweb.org/anthology/attachments/D/D16/D16-1019.Attachment.zip
$ unzip D16-1019.Attachment.zip
$ cd ..
$ mkdir clauses
$ cat data/wb18/wn18_rules | tr -d "\r" | sed -e ''s/"(x,y)"/"(X,Y)"/g'' | sed -e ''s/"(y,x)"/"(Y,X)"/g'' | sed -e ''s/"(x,z)"/"(X,Z)"/g'' | sed -e ''s/"(y,z)"/"(Y,Z)"/g'' | sed -e ''s/"(z,x)"/"(Z,X)"/g'' | sed -e ''s/"(z,y)"/"(Z,Y)"/g'' | sed -e ''s/"==>"/" :- "/g'' | awk '{ print $3 " :- " $1 }'  | sed -e ''s/"&&"/", "/g'' > clauses/wn18-clauses.pl
$ cat data/fb122/fb122_rules | tr -d "\r" | sed -e ''s/"(x,y)"/"(X,Y)"/g'' | sed -e ''s/"(y,x)"/"(Y,X)"/g'' | sed -e ''s/"(x,z)"/"(X,Z)"/g'' | sed -e ''s/"(y,z)"/"(Y,Z)"/g'' | sed -e ''s/"(z,x)"/"(Z,X)"/g'' | sed -e ''s/"(z,y)"/"(Z,Y)"/g'' | sed -e ''s/"==>"/" :- "/g'' | awk '{ print $3 " :- " $1 }'  | sed -e ''s/"&&"/", "/g'' > clauses/fb122-clauses.pl
```
