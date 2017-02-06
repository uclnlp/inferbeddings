# NYT

```bash
$ wget -O naacl2013.txt.zip https://www.dropbox.com/s/5iulumlihydo1k7/naacl2013.txt.zip?dl=1
$ unzip naacl2013.txt.zip
$ rm -f naacl2013.txt.zip
```

Creation of the dataset:

```bash
$ cat naacl2013.txt | awk -F '\t' '{ if ($4 == "Train") print $2 "\t" $1 "\t" $3 "\t" $5 }' > naacl2013_train.tsv
$ cat naacl2013.txt | awk -F '\t' '{ if ($4 == "Test") print $2 "\t" $1 "\t" $3 "\t" $5 }' > naacl2013_test.tsv
$ wc -l *.tsv
  104448 naacl2013_test.tsv
  118781 naacl2013_train.tsv
  223229 total
$ md5sum *.tsv
e98f69e22b192cf54ae7882ce1cf2ec4  naacl2013_test.tsv
d4854bdc431ed7bc90a8cdbfa60a174a  naacl2013_train.tsv
```