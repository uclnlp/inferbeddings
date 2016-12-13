cat uml.db  | tr "(" "\t" | tr "," "\t" | tr -d ")" | awk '{ print $3 " " $2 " " $4 }' > umls.tsv
