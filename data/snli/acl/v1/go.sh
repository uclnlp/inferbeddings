./tools/sample.py -p ../../snli_1.0_test.jsonl.gz -f 0.01 | ./tools/invert.py | jq . | tr -d "\n" | sed -e ''s/"}"/"}\n"/g'' | jq . > start.json
