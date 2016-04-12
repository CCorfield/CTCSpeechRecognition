#!/bin/bash

dir="$1VoxForgeAudio/"
echo "Downloading Voxforge Corpus to: " ${dir}
echo "The dataset is roughly 16gb. This will take time to download depending on your internet speeds."
wget -r -nH -nd -np -R index.html* -P ${dir} http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/
echo "Download completed. Extracting files..."
for tgz in "$dir"/*
	do
	    echo "$tgz"
	    tar zxvf "$tgz" -C "$dir"
	    rm -r "$tgz"
	done
echo "Finished, corpus has been downloaded to " ${dir}