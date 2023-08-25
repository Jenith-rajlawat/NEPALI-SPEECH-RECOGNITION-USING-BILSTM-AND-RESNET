# NEPALI-SPEECH-RECOGNITION-USING-BILSTM-AND-RESNET
## Keywords
```Speech To Text, Nepali, CNN, ResNet, BiLSTM, CTC ```
## Intorduction
This repo is a part of the research project for designing the automatic speech recogntion(ASR) model for Nepali language using ML techniques. 

## Things to consider before
- You are free to use this research as a reference and make modifications to continue your own research in Nepali ASR. 
- The `trainer.py` has been implemented to run on the sampled data for now. To replicate the result please replace dataset directory with original [OpenSLR dataset.](https://openslr.org/54)
- Please remove the (audio, text) pairs that include Devnagari numeric texts like १४२३, ५९२, etc from the dataset because they degrade the performance of the model.
  
## Our approach
0. Remove the (audio, text) pairs that include Devnagari numeric transcriptions
1. Data cleaning (clipping silent gaps from both ends)
2. MFCC feature extraction from audio data
3. Design Neural Network (optimal: CNN + ResNet + BiLSTM) model 
4. Calculate CTC loss for applying gradient (training)
5. Decode the texts by using beam search decoding (infernce)


## Running the project
0. Initialize the virtual environment by installing packages from `requirements.txt`.
1. Run the training pipeline & evaluate authors model, which can be also be used to evaluate your own (audio,text) pairs.
```
python trainer.py   # For running the training pipeline
python eval.py      # For testing and evaluating the model already trained by the author
```


