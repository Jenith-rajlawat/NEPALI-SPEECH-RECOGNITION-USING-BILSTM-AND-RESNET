# You can find all of the necessary functions properly documented in utils.py

from model.configs import UNQ_CHARS
from model.utils import CER_from_wavs, ctc_softmax_output_from_wavs, load_model, load_wav, plot_losses, predict_from_wavs


if __name__ == "__main__":

    # Loads the trained model
    print("Loading model.....")
    model = load_model("model/trained_model1.h5")
    print("Model loaded \u2705 \u2705 \u2705 \u2705\n")
    

    # Loads wav file
    wavs = []
    print("Loading wav files.....")
    wavs.append(load_wav("dataset/wav_files(sampled)/0f43e91c4e.flac"))
    wavs.append(load_wav("dataset/wav_files(sampled)/0f6725b07e.flac"))
    print("Wav files loaded \u2705 \u2705 \u2705 \u2705\n")
    
    """Gives the array of predicted sentences"""
    print("Predicting sentences.....")
    sentences, char_indices = predict_from_wavs(model, wavs, UNQ_CHARS)
    print(sentences, "\n")

    """Gives softmax output of the ctc model"""
    # softmax = ctc_softmax_output_from_wavs(model, [wav])
    # print(softmax)

    """Gives Character Error Rate (CER) between the targeted and predicted output"""
    print("Calculating CER.....")
    cer = CER_from_wavs(model, wavs, ["राज्य सिक्किमको", "लगेर हानेमा गलत"], UNQ_CHARS)
    print(cer, "\n")