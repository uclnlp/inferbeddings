# Countries

```
$ mkdir data
$ wget https://github.com/mledoze/countries/raw/master/dist/countries.csv
$ md5sum countries.csv 
4f68eaef5b591ca0c0d70878448b2f93  countries.csv
$ ./countries.py
$ md5sum countries.tsv data/*.lst
e7107a9b52f67ba12180698a1a931a96  countries.tsv
0180b728946f110c5f46a5c20673e6f7  data/countries.lst
00befb2849fc2ed7f9fc8e8573c14fdd  data/regions.lst
7a9b287c75075eff42cc5b09fd0764c5  data/subregions.lst
```
