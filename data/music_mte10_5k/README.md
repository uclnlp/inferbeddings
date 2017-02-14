# DBpedia 2015-10 Music, MTE10

#### Download

Data downloaded on 14/07/2016 by using the following commands:

```bash
pasquale@koeln:music_2015-10_mte10_5k$ mkdir download
pasquale@koeln:music_2015-10_mte10_5k$ cd download/
pasquale@koeln:download$ wget -c http://downloads.dbpedia.org/2015-10/core-i18n/en/mappingbased_objects_en.ttl.bz2
--2016-07-14 15:30:50--  http://downloads.dbpedia.org/2015-10/core-i18n/en/mappingbased_objects_en.ttl.bz2
Resolving downloads.dbpedia.org (downloads.dbpedia.org)... 139.18.16.66
Connecting to downloads.dbpedia.org (downloads.dbpedia.org)|139.18.16.66|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 162090333 (155M) [application/octet-stream]
Saving to: ‘mappingbased_objects_en.ttl.bz2’
[..]
pasquale@koeln:download$ md5sum mappingbased_objects_en.ttl.bz2 
953e171dd48ac772147789c1bbc18fb9  mappingbased_objects_en.ttl.bz2
```

#### Filtering

We need first to filter the desired predicates, and then to enforce the MTE10 limit.

For filtering the desired predicates, we do the following, creating a TSV file creating all Music triples:

```bash
pasquale@koeln:download$ bzcat mappingbased_objects_en.ttl.bz2 | grep '<http://dbpedia.org/ontology/genre>\|<http://dbpedia.org/ontology/recordLabel>\|<http://dbpedia.org/ontology/associatedMusicalArtist>\|<http://dbpedia.org/ontology/associatedBand>\|<http://dbpedia.org/ontology/musicalArtist>\|<http://dbpedia.org/ontology/musicalBand>\|<http://dbpedia.org/ontology/album>' | sed -e ''s/"> <"/">\t<"/g'' | sed -e ''s/"> ."/">"/g'' > ../music_2015-10.nt
pasquale@koeln:download$ md5sum ../music_2015-10.nt
23196cd7f04a11c31e2c33a1cfe68ee6  ../music_2015-10.nt
```

Then, we create a list of all entities occurring at least 10 times:

```bash
pasquale@koeln:download$ cd ..
pasquale@koeln:music_2015-10_mte10_5k$ 
pasquale@koeln:music_2015-10_mte10_5k$ mkdir stats
pasquale@koeln:music_2015-10_mte10_5k$ cat music_2015-10.nt | awk -F "\t" '{print $1 "\n" $3}' | sort | uniq -c | awk '{if ($1 >= 10) {print $2}}' > stats/music_2015-10_entities_mte10.txt
pasquale@koeln:music_2015-10_mte10_5k$ md5sum stats/music_2015-10_entities_mte10.txt 
ef8e89f2a332a9b8b3c5cac2a2012cfc  stats/music_2015-10_entities_mte10.txt
pasquale@koeln:music_2015-10_mte10_5k$ wc -l stats/music_2015-10_entities_mte10.txt 
33347 stats/music_2015-10_entities_mte10.txt
```

Now we create the MTE10 version of the Knowledge Graph:

```bash
pasquale@koeln:music_2015-10_mte10_5k$ ./tools/filter.py music_2015-10.nt stats/music_2015-10_entities_mte10.txt > music_2015-10_mte10.nt 
INFO:root:Acquiring music_2015-10.nt ..
pasquale@koeln:music_2015-10_mte10_5k$ md5sum music_2015-10_mte10.nt 
db6fd7235b88c5d37eada5c5e5077423  music_2015-10_mte10.nt
pasquale@koeln:music_2015-10_mte10_5k$ wc -l music_2015-10_mte10.nt 
299825 music_2015-10_mte10.nt
pasquale@koeln:music_2015-10_mte10_5k$ wc -l *.nt
   299825 music_2015-10_mte10.nt
  1214352 music_2015-10.nt
  1514177 total
pasquale@koeln:music_2015-10_mte10_5k$ gzip -9 music_2015-10*.nt
pasquale@koeln:music_2015-10_mte10_5k$ du -hs *.gz
2.6M	music_2015-10_mte10.nt.gz
13M	music_2015-10.nt.gz
```

#### Splitting in Training, Validation and Test sets

Now we split the filtered Knowledge Graph `music_2015-10_mte10.nt` into training, validation and test sets:

```bash
pasquale@koeln:music_2015-10_mte10_5k$ ./tools/split.py music_2015-10_mte10.nt.gz --train music_2015-10_mte10-train.tsv --valid music_2015-10_mte10-valid.tsv --valid-size 5000 --test music_2015-10_mte10-test.tsv --test-size 5000
DEBUG:root:Importing the Knowledge Graph ..
DEBUG:root:Number of triples in the Knowledge Graph: 299825
DEBUG:root:Generating a random permutation of RDF triples ..
DEBUG:root:Building the training, validation and test sets ..
DEBUG:root:Saving ..
pasquale@koeln:music_2015-10_mte10_5k$ wc -l *.tsv
    5000 music_2015-10_mte10-test.tsv
  289825 music_2015-10_mte10-train.tsv
    5000 music_2015-10_mte10-valid.tsv
  299825 total
pasquale@koeln:music_2015-10_mte10_5k$ zcat music_2015-10_mte10.nt.gz | wc -l
299825
pasquale@koeln:music_2015-10_mte10_5k$ md5sum *.tsv
718e8a82e7df01728ee94b30ec06d9f8  music_2015-10_mte10-test.tsv
f14e0fb96ad1eca43d5dfbec3316918f  music_2015-10_mte10-train.tsv
4e343c10c7b97999c1b4eff90fb11090  music_2015-10_mte10-valid.tsv
pasquale@koeln:music_2015-10_mte10_5k$ gzip -9 *.tsv
```