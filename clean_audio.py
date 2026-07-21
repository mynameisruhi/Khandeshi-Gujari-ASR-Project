from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import numpy as np
from scipy.io import wavfile
from IPython.display import Audio

original_video =  # file name goes here


if __name__ == "__main__":
    model, df_state, _ = init_df()
    audio_path = original_video
    audio, _ = load_audio(audio_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio("enhanced.wav", enhanced, df_state.sr())

sample_rate, data = wavfile.read('enhanced.wav')
