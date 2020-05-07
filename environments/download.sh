#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: download.sh <environment name>"
    echo "Environments: banana, reacher"
    exit
fi

DIRECTORY=$(cd `dirname $0` && pwd)
TMPFILE=`mktemp`

case $1 in

    banana)
        echo "Downloading BananaCollector environment into $DIRECTORY/Banana_Linux/"
        wget --no-hsts https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip -O $TMPFILE
        unzip -q -d $DIRECTORY $TMPFILE
        rm $TMPFILE
        ;;
    reacher)
        echo "Downloading Reacher environment into $DIRECTORY/Reacher_Linux/"
        wget --no-hsts  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip -O $TMPFILE
        unzip -q -d $DIRECTORY $TMPFILE
        rm $TMPFILE
        ;;
    *)
        echo "unknown environment name"
        ;;
esac
