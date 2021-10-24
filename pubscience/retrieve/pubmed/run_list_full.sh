#!/bin/bash
while read p; do
	#echo $p | fold -w 100 -s | paste -sd _
	filename=""
	for w in $p; do
		filename=${filename}_${w}
	done
	filename=${filename#"_"}
	echo $p
	echo "|" | esearch -db pmc -query "\"$p\"" | efetch -start 0 -stop 1000 -format xml > $root_folder/pubmed/$xml_dump_folder/full_$filename.xml
done < searchterms.txt
