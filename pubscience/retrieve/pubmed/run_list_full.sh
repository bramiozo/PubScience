#!/bin/bash
while read p; do
	#echo $p | fold -w 100 -s | paste -sd _
	filename=""
	for w in $p; do
		filename=${filename}_${w}
	done
	filename=${filename#"_"}
	echo $p
	echo "|" | esearch -db pmc -query "\"$p\"" | efetch -format xml > $root_folder/$xml_dump_folder/full_$filename.xml
done < searchterms5.txt
