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
        with request.urlopen(url) as response:
            with open(filename, 'wb') as fb:
                shutil.copyfileobj(response, fb)
    return filename


def transcribe(
    model_name,
    audio_id,
    **kwargs,
):
    model = whisper.load_model(model_name)
    filename = download_file(audio_id)

    start_time = time.time()
    result = model.transcribe(filename, verbose=None, **kwargs)
    end_time = time.time()

    result.update(
        start_time=start_time,
        end_time=end_time,
        elapsed=end_time - start_time,
    )
    return result
