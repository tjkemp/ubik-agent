#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: download.sh <environment name>"
    echo "Environments: banana, visualbanana, reacher, crawler"
    exit
fi

ENVIRONMENTS_DIR=environments

CURRENT_DIR=$(cd `dirname $0` && pwd)
DESTINATION=$CURRENT_DIR/$ENVIRONMENTS_DIR

mkdir -p $DESTINATION

TMPFILE=`mktemp`

case $1 in
    banana)
        echo "Downloading BananaCollector environment into $DESTINATION/Banana_Linux/"
        wget --no-hsts https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip -O $TMPFILE
        unzip -q -d $DESTINATION $TMPFILE
        rm $TMPFILE
        ;;
    visualbanana)
        echo "Downloading VisualBanana environment into $DESTINATION/VisualBanana_Linux/"
        wget --no-hsts https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip -O $TMPFILE
        unzip -q -d $DESTINATION $TMPFILE
        rm $TMPFILE
        ;;
    reacher)
        echo "Downloading Reacher environment into $DESTINATION/Reacher_Linux/"
        wget --no-hsts  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip -O $TMPFILE
        unzip -q -d $DESTINATION $TMPFILE
        rm $TMPFILE
        ;;
    crawler)
        echo "Downloading Crawler environment into $DESTINATION/Crawler_Linux/"
        wget --no-hsts  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip -O $TMPFILE
        unzip -q -d $DESTINATION $TMPFILE
        rm $TMPFILE
        ;;
    *)
        echo "unknown environment name"
        ;;
esac
