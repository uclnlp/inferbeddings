# Countries

```
$ mkdir data
$ wget https://github.com/mledoze/countries/raw/master/dist/countries.csv
$ md5sum countries.csv 
4f68eaef5b591ca0c0d70878448b2f93  countries.csv
$ ./countries.py
$ md5sum countries.tsv data/*.lst
2b281a7522074b5aba704dd93a31b706  countries.tsv
f42f12a5e5140b27fef1a8087c1ae0c7  data/countries.lst
655ca53f0f64733e1349968a473079cd  data/regions.lst
69e4f8cb63bab522ea8954ec848c0712  data/subregions.lst
```
