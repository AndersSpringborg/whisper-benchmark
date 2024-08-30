import os
import time
from urllib import request
import shutil
import whisper
from whisper_benchmark import constants


def download_file(name):
    url = constants.AUDIO_FILES[name]['url']
    filename = os.path.join(constants.CACHE_DIR, f"{name}.{url.split('.')[-1]}")
    os.makedirs(constants.CACHE_DIR, exist_ok=True)
    if not os.path.exists(filename):
        request.urlretrieve(url, filename)
    return filename


def transcribe(
    model_name,
    audio_id,
    device,
    **kwargs,
):
    print("loading model")
    model = whisper.load_model(model_name, device=device)
    filename = download_file(audio_id)
    audio = whisper.load_audio(filename)
    language = audio_id.split('-')[0]
    kwargs.setdefault('language', language)

    print("Starting transcription")
    start_time = time.time()
    result = model.transcribe(audio, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    print("Ending transcripton. Took" + str(elapsed))

    result.pop('segments')
    result.update(
        start_time=start_time,
        end_time=end_time,
        elapsed=elapsed,
        fps=10,
        device=device,
        language=kwargs['language'],
    )
    return result
