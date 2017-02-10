# Embeddings

```bash
./bin/adv-cli.py --train data/synth/transitive-tiny/data.tsv --clauses data/synth/transitive-tiny/clauses.pl --model TransE --similarity l1 --margin 1 --embedding-size 6 --nb-epochs 100 --adv-lr 0.1 --nb-batches 1 --adv-weight 1000 --debug-embeddings models/visualize/transitivity --adv-batch-size 1
```
