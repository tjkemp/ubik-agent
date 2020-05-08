#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: download.sh <environment name>"
    echo "Environments: banana, visualbanana, reacher, crawler"
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
    visualbanana)
        echo "Downloading VisualBanana environment into $DIRECTORY/VisualBanana_Linux/"
        wget --no-hsts https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip -O $TMPFILE
        unzip -q -d $DIRECTORY $TMPFILE
        rm $TMPFILE
        ;;
    reacher)
        echo "Downloading Reacher environment into $DIRECTORY/Reacher_Linux/"
        wget --no-hsts  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip -O $TMPFILE
        unzip -q -d $DIRECTORY $TMPFILE
        rm $TMPFILE
        ;;
    crawler)
        echo "Downloading Crawler environment into $DIRECTORY/Crawler_Linux/"
        wget --no-hsts  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip -O $TMPFILE
        unzip -q -d $DIRECTORY $TMPFILE
        rm $TMPFILE
        ;;
    *)
        echo "unknown environment name"
        ;;
esac
