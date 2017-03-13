### Jointly Embedding Knowledge Graphs and Logical Rules

```
$ mkdir data ; cd data
$ wget -c http://www.aclweb.org/anthology/attachments/D/D16/D16-1019.Attachment.zip
$ unzip D16-1019.Attachment.zip
$ cd ..
$ mkdir -p wn18/clauses/
$ cp data/wb18/wn18.triples.train data/wb18/wn18.triples.valid data/wb18/wn18.triples.test wn18/
$ cat data/wb18/wn18_rules | tr -d "\r" | sed -e ''s/"(x,y)"/"(X,Y)"/g'' | sed -e ''s/"(y,x)"/"(Y,X)"/g'' | sed -e ''s/"(x,z)"/"(X,Z)"/g'' | sed -e ''s/"(y,z)"/"(Y,Z)"/g'' | sed -e ''s/"(z,x)"/"(Z,X)"/g'' | sed -e ''s/"(z,y)"/"(Z,Y)"/g'' | sed -e ''s/"==>"/" :- "/g'' | awk '{ print $3 " :- " $1 }'  | sed -e ''s/"&&"/", "/g'' > wn18/clauses/wn18-clauses.pl
$ mkdir fb122/clauses/
$ cp data/fb122/fb122_triples.train data/fb122/fb122_triples.valid data/fb122/fb122_triples.test fb122/
$ cat data/fb122/fb122_rules | tr -d "\r" | sed -e ''s/"(x,y)"/"(X,Y)"/g'' | sed -e ''s/"(y,x)"/"(Y,X)"/g'' | sed -e ''s/"(x,z)"/"(X,Z)"/g'' | sed -e ''s/"(y,z)"/"(Y,Z)"/g'' | sed -e ''s/"(z,x)"/"(Z,X)"/g'' | sed -e ''s/"(z,y)"/"(Z,Y)"/g'' | sed -e ''s/"==>"/" :- "/g'' | awk '{ print $3 " :- " $1 }'  | sed -e ''s/"&&"/", "/g'' > fb122/clauses/fb122-clauses.pl
```
