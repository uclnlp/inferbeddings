./tools/sample.py -p ../../snli_1.0_test.jsonl.gz -f 0.01 | ./tools/invert.py | jq . | tr -d "\n" | sed -e ''s/"}"/"}\n"/g'' | jq . > v1.1_start.json
./tools/genadv_dam.py -p ../../snli_1.0_test.jsonl.gz -n 100 | jq . > v1.2_start.json

cat v1.1_edited.json | tr -d "\n" | sed -e ''s/"}"/"}\n"/g'' > v1.1_edited.jsonl
cat v1.2_edited.json | tr -d "\n" | sed -e ''s/"}}"/"}}\n"/g'' > v1.2_edited.jsonl
