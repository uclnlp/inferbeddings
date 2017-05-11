from analysis.ERMLP.ermlp import *

train_triples = read_triples('wn18.triples.train')
valid_triples = read_triples('wn18.triples.valid')
test_triples = read_triples('wn18.triples.test')

all_triples = train_triples + valid_triples + test_triples
entity_set = set([s for (s, p, o) in all_triples] + [o for (s, p, o) in all_triples])
predicate_set = set([p for (s, p, o) in all_triples])

nb_entities, nb_predicates = len(entity_set), len(predicate_set)

forward_mapping = {
    "_member_of_domain_topic": "_synset_domain_topic_of",
    "_synset_domain_usage_of": "_member_of_domain_usage",
    "_instance_hyponym": "_instance_hypernym",
    "_hyponym": "_hypernym",
    "_member_holonym": "_member_meronym",
    "_synset_domain_region_of": "_member_of_domain_region",
    "_part_of": "_has_part",
    "_member_meronym": "_member_holonym",
    "_hypernym": "_hyponym",
    "_synset_domain_topic_of": "_member_of_domain_topic",
    "_instance_hypernym": "_instance_hyponym",
    "_has_part": "_part_of",
    "_member_of_domain_region": "_synset_domain_region_of",
    "_member_of_domain_usage": "_synset_domain_usage_of",
    "_derivationally_related_form": "_derivationally_related_form",
    "_verb_group": "_verb_group"
}
mapping = dict(list(forward_mapping.items()) + [(value, key) for key, value in forward_mapping.items()])

print(train_triples[:10])
print(mapping)

train_set = set(train_triples)
err_corrupt_subj, err_corrupt_obj = [], []
for s, p, o in valid_triples:
    if p not in mapping:
        print(p)
    if p in mapping and (o, mapping[p], s) in train_set:
        err_corrupt_obj += [1]
        err_corrupt_subj += [1]
    else:
        err_corrupt_obj += [nb_entities]
        err_corrupt_subj += [nb_entities]

err = err_corrupt_subj + err_corrupt_obj
print(err)
print('Mean Rank: {}'.format(np.mean(err)))

for k in [1, 3, 5, 10]:
    print('Hits@{}: {}'.format(k, np.mean(np.asarray(err) <= k) * 100))

