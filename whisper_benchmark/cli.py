import sys
import argparse
import warnings

import numpy as np
import numba
import torch

import whisper
from whisper import utils

import whisper_benchmark
from whisper_benchmark import run

MODEL_NAMES = (
    'tiny',
    'base',
    'small',
    'medium',
    'large',
    'large-v1',
    'large-v2',
    'large-v3',
    'tiny.en',
    'base.en',
    'small.en',
    'medium.en',
)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_id", type=str, help="audio file to transcribe")
    parser.add_argument("--model-name", default="small", choices=MODEL_NAMES, help="name of the Whisper model to use")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")

    parser.add_argument("--verbose", type=int, default=0, help="0: Muted, 1: Info, 2: Verbose")
    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=utils.optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=utils.optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", default=False, action="store_true", help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", default=False, action="store_true", help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=utils.optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=utils.optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=utils.optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=utils.optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    opts = vars(parser.parse_args())

    temperature = opts.pop("temperature")
    if (increment := opts.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]
    opts['temperature'] = temperature

    threads = opts.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)
    else:
        threads = torch.get_num_threads()

    if opts['verbose'] == 0:
        opts['verbose'] = None
        warnings.filterwarnings("ignore")
    elif opts['verbose'] == 1:
        opts['verbose'] = None
    elif opts['verbose'] == 2:
        opts['verbose'] = False
    elif opts['verbose'] > 2:
        opts['verbose'] = True

    result = run.transcribe(**opts)

    text = result.pop('text')
    if (opts['verbose'] or 0) > 2:
        sys.stderr.write(text)

    result.update(
        model=opts['model_name'],
        audio_id=opts['audio_id'],
        version=whisper_benchmark.__version__,
        torch_version=torch.version.__version__,
        cuda_version=torch.version.cuda,
        python_version=sys.version.split()[0],
        whisper_version=whisper.version.__version__,
        numba_version=numba.version_info.string,
        numpy_version=np.version.version,
        threads=threads,
    )
    for key, value in result.items():
        print(key, ':', value)


if __name__ == '__main__':
    main()
