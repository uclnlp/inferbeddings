# DBpedia 2015-10 Music, MTE10

#### Download

Data downloaded on 14/07/2016 by using the following commands:

```bash
$ mkdir download
$ cd download/
$ wget -c http://downloads.dbpedia.org/2015-10/core-i18n/en/mappingbased_objects_en.ttl.bz2
--2016-07-14 15:30:50--  http://downloads.dbpedia.org/2015-10/core-i18n/en/mappingbased_objects_en.ttl.bz2
Resolving downloads.dbpedia.org (downloads.dbpedia.org)... 139.18.16.66
Connecting to downloads.dbpedia.org (downloads.dbpedia.org)|139.18.16.66|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 162090333 (155M) [application/octet-stream]
Saving to: ‘mappingbased_objects_en.ttl.bz2’
[..]
$ md5sum mappingbased_objects_en.ttl.bz2 
953e171dd48ac772147789c1bbc18fb9  mappingbased_objects_en.ttl.bz2
```

#### Filtering

We need first to filter the desired predicates, and then to enforce the MTE10 limit.

For filtering the desired predicates, we do the following, creating a TSV file creating all Music triples:

```bash
$ bzcat mappingbased_objects_en.ttl.bz2 | grep '<http://dbpedia.org/ontology/genre>\|<http://dbpedia.org/ontology/recordLabel>\|<http://dbpedia.org/ontology/associatedMusicalArtist>\|<http://dbpedia.org/ontology/associatedBand>\|<http://dbpedia.org/ontology/musicalArtist>\|<http://dbpedia.org/ontology/musicalBand>\|<http://dbpedia.org/ontology/album>' | sed -e ''s/"> <"/">\t<"/g'' | sed -e ''s/"> ."/">"/g'' > ../music_2015-10.nt
$ md5sum ../music_2015-10.nt
23196cd7f04a11c31e2c33a1cfe68ee6  ../music_2015-10.nt
$ cat ../music_2015-10.nt | awk '{ print $1 "\t\"" $2 "\"\t" $3 }' > ../music.nt
d5be653a784a14571f17835d52f6f075  music.nt
$ gzip -9 ../*.nt
```

Then, we create a list of all entities occurring at least 10 times:

```bash
download$ cd ..
$ 
$ mkdir stats
$ zcat music.nt.gz | awk -F "\t" '{print $1 "\n" $3}' | sort | uniq -c | awk '{if ($1 >= 10) {print $2}}' > stats/music_entities_mte10.txt
$ md5sum stats/music_entities_mte10.txt 
ef8e89f2a332a9b8b3c5cac2a2012cfc  stats/music_entities_mte10.txt
$ wc -l stats/music_entities_mte10.txt 
33347 stats/music_entities_mte10.txt
```

Now we create the MTE10 version of the Knowledge Graph:

```bash
$ gunzip music.nt.gz
$ ./tools/filter.py music.nt stats/music_entities_mte10.txt > music_mte10.nt
INFO:root:Acquiring music.nt ..
$ md5sum music_mte10.nt 
55b61f3116f95c52be693293b3600030  music_mte10.nt
$ wc -l music_mte10.nt 
299825 music_mte10.nt
$ wc -l *.nt
   299825 music_mte10.nt
  1214352 music.nt
  1514177 total
$ gzip -9 *.nt
```

#### Splitting in Training, Validation and Test sets

Now we split the filtered Knowledge Graph `music_2015-10_mte10.nt` into training, validation and test sets:

```bash
$ ./tools/split.py music_mte10.nt.gz --train music_mte10-train.tsv --valid music_mte10-valid.tsv --valid-size 5000 --test music_mte10-test.tsv --test-size 5000
DEBUG:root:Importing the Knowledge Graph ..
299825it [00:01, 285777.47it/s]
DEBUG:root:Number of triples in the Knowledge Graph: 299825
DEBUG:root:Generating a random permutation of RDF triples ..
DEBUG:root:Building the training, validation and test sets ..
DEBUG:root:Saving ..
$ wc -l *.tsv
    5000 music_mte10-test.tsv
  289825 music_mte10-train.tsv
    5000 music_mte10-valid.tsv
  299825 total
$ zcat music_mte10.nt.gz | wc -l
299825
$ md5sum *.tsv
2c503c8bd4f6a1225e4cd0402b225fff  music_mte10-test.tsv
f7701211f305c25b62fe1f571057f697  music_mte10-train.tsv
71182c8a8648f683eca3c452dfeac0d7  music_mte10-valid.tsv
$ gzip -9 *.tsv
```
