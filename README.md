# CTCSpeechRecognition
Speech Recognition using DeepSpeech2 network and the CTC activation function.

Snapshot as of 29 May of the phoneme version of CTC recognition under SeanNaren/CTCSpeechRecognition. This does not include any of Seed93's multi-GPU changes.

These scripts assume that the audio, transcripts, phonemes have been preprocessed into Torch 7 files, which are tables with three members "text", "input", and "labels".

The first part of corpus processing is to convert the audio to .wav files, the second part is to convert the training and testing files to Torch files, see AN4CTCorpus. 

The AN4CTCTrain.lua script can used any of the DeepSpeech*.lua models provided (look at the top of the training file).

You will need to update the data locations referenced in the Lua file to your own organization.
