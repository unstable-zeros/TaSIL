#!/usr/bin/env bash
PROJECT=sil

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cp $DIR/../requirements.txt $DIR/requirements.txt.tmp
docker build -t $USER-$PROJECT-container $DIR
