#!/bin/bash 
while read p; do
	#echo $p | fold -w 100 -s | paste -sd _
	filename=""
	for w in $p; do
		filename=${filename}_${w}
	done
	filename=${filename#"_"}
	echo "searchterm:" $p
	echo "|" | esearch -db pubmed -query "\"$p\"" | efetch -format abstract -mode xml > $root_folder/pubmed/$xml_dump_folder/abstract_$filename.xml
done < searchterms.txt
