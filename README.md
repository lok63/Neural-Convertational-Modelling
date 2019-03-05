# Neural-Convertational-Modelling
Chatbot developed in TensorFlow. Seq2seq vanilla, Attention and Beam Serch

## Overview
A chatbot trained on the Cornell Movie-Dialogs Corpus and evaluated using BLEU-4 and perplexity. Other methods such as Attention and Beam search were used to augment the model. Our chatbot is able to generate human like responses. 

## TO-DO
- [ ] Pre-processing
- [ ] Use MMI as loss function to solve the problem of common responses
- [ ] Create dictionaries to map users for remembering conversations

#### Train the model
Run:
```
python train.py
```
All weights will be stored in the checkpoints directory. If the directory is empty a new model will be created. Otherwise the model will use the most recent wights to continue training from the last checkpoint. During training make sure BeamSearch is set to False. The default hyperparameters were set in the /config/ file. The default choice is Attention but it can be set to False in order to train using the baseline Seq2seq model.

#### Predict
Run:
```
python predict.py
```
BeamSearch can be set to True here if you want multipble responses.

