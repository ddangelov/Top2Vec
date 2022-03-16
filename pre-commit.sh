#!/bin/sh

if git rev-parse --verify HEAD >/dev/null 2>&1
then
	against=HEAD
else
	# Initial commit: diff against an empty tree object
	against=$(git hash-object -t tree /dev/null)
fi

# If you want to allow non-ASCII filenames set this variable to true.
allownonascii=$(git config --type=bool hooks.allownonascii)

BLACK_NB_ERRORS="Good"
# Make sure that any notebooks don't have output
files=$(git diff --cached $against --name-only | sort | uniq | grep .ipynb)
echo $files
for file in $files; do
	if [ "$file" = "" -o ! -f "$file" ]; then
		continue
	fi
	echo $file
	black-nb -o $file
    if [ $? -ne 0 ]; then
        BLACK_NB_ERRORS="OHNO"
	else
	    echo "$file cleaned"
	fi
done

if [ $BLACK_NB_ERRORS != "Good" ]; then
    echo "Problem with notebooks in pre-commit."
    exit 1
fi
