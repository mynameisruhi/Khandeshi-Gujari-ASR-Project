from elevenlabs import ElevenLabs

api_key =   # insert API key here

elevenlabs = ElevenLabs(api_key=api_key)

def transcribe_multichannel(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        result = elevenlabs.speech_to_text.convert(
            file=audio_file,
            model_id='scribe_v2',
            use_multi_channel=True,
            diarize=False,
            timestamps_granularity='word'
        )
    return result

result = transcribe_multichannel('audio.wav')

print(result.text)
