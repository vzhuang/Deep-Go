#!/bin/bash
#
for i in {1..2000};
do
	wget -I /f -r -H -nd -nc -A.sgf http://gokifu.com/?p=$i
done
