yes | head -n 8192 | awk '{ print "e" i++ "\tp\te" i++ }' > data.tsv
printf "a\tq\tb\nc\tr\td\ne\ts\tf\ng\tt\th\n" >> data.tsv

yes | head -n 8192 | awk '{ print "e" i++ "\tt\te" i++ }' > data-test.tsv

