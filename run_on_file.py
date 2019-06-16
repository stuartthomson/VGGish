from vggish import VGGish
from scipy.io import wavfile
from numpy.random import seed, randint

from preprocess_sound import preprocess_sound
import numpy as np

from keras.models import Model
from keras.layers import GlobalAveragePooling2D


def main():
    sound_model = VGGish(include_top=False)

    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    audio_file = "test.wav"
    sr, wav_data = wavfile.read(audio_file)
    print(sr, wav_data)

    seg_len = 3 # 3s
    seg_num = 1
    sample_num = 1

    data = np.zeros((496, 64, 1))
    label = [0]
    
    length = sr * seg_len           # 3s segment
    range_high = len(wav_data) - length
    seed(1)  # for consistency and replication
    random_start = randint(range_high, size=seg_num)

    i = 0
    
    # for j in range(seg_num):
    cur_wav = wav_data

    print('cur_wav:', cur_wav)

    cur_wav = cur_wav / 32768.0
    cur_spectro = preprocess_sound(cur_wav, sr)

    print('spectro:', cur_spectro)

    cur_spectro = np.expand_dims(cur_spectro, 3)
    data = cur_spectro

    print(data, sound_extractor)

    predictions = sound_extractor.predict(data)

    return predictions, len(predictions)

if __name__ == "__main__":
    print(main())
    