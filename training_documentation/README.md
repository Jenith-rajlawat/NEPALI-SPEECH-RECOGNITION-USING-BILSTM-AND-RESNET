# NEPALI-SPEECH-RECOGNITION-USING-BILSTM-AND-RESNET

## Epoch and losses
  -(https://openslr.org/54) Dataset and the zip mentioned in this folder can be seen in this website
  -We are observing the training and testing losses for different epoch and batch size 
  -Each zip is extracted and the audio is collected in one folder, plus a file,speaker,transcription file (.csv) created just for that zip 
  - Model is trained for each zip subsequently. 
  -This folder shows the progress for each zip

  
## Our approach
0. Remove the (audio, text) pairs that include Devnagari numeric transcriptions
1. Data cleaning (clipping silent gaps from both ends)
2. MFCC feature extraction from audio data
3. Design Neural Network (optimal: CNN + ResNet + BiLSTM) model 
4. Calculate CTC loss for applying gradient (training)
5. Decode the texts by using beam search decoding (infernce)



