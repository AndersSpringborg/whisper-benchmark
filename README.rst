Whisper Benchmark
========================

**Whisper-Benchmark** is a simple tool to evaluate performance of `Whisper <https://github.com/openai/whisper>`_ models and configurations.

Install
-------

.. note::

  This project must be currently coupled with the `Cloud Mercato's version <https://github.com/cloudmercato/whisper>`_ of Whisper.
  A `pull request <https://github.com/openai/whisper/pull/1787>`_ is in progress about that.

After installing Whisper: ::

  pip install https://github.com/cloudmercato/whisper-benchmark/archive/refs/heads/master.zip
  
  
Usage
-----

Command line
~~~~~~~~~~~~

Most of the original Whisper options are available:::

  $ whisper-benchmark --help
  usage: whisper-benchmark [-h]
                           [--model-name {tiny,base,small,medium,large,large-v1,large-v2,large-v3,tiny.en,base.en,small.en,medium.en}]
                           [--device DEVICE] [--task {transcribe,translate}] [--verbose VERBOSE]
                           [--temperature TEMPERATURE] [--best_of BEST_OF] [--beam_size BEAM_SIZE]
                           [--patience PATIENCE] [--length_penalty LENGTH_PENALTY]
                           [--suppress_tokens SUPPRESS_TOKENS] [--initial_prompt INITIAL_PROMPT]
                           [--condition_on_previous_text] [--fp16]
                           [--temperature_increment_on_fallback TEMPERATURE_INCREMENT_ON_FALLBACK]
                           [--compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD]
                           [--logprob_threshold LOGPROB_THRESHOLD]
                           [--no_speech_threshold NO_SPEECH_THRESHOLD] [--threads THREADS]
                           audio_id

  positional arguments:
    audio_id              audio file to transcribe

  options:
    -h, --help            show this help message and exit
    --model-name {tiny,base,small,medium,large,large-v1,large-v2,large-v3,tiny.en,base.en,small.en,medium.en}
                          name of the Whisper model to use (default: small)
    --device DEVICE       device to use for PyTorch inference (default: cpu)
    --task {transcribe,translate}
                          whether to perform X->X speech recognition ('transcribe') or X->English
                          translation ('translate') (default: transcribe)
    --verbose VERBOSE     0: Muted, 1: Info, 2: Verbose (default: 0)
    --temperature TEMPERATURE
                          temperature to use for sampling (default: 0)
    --best_of BEST_OF     number of candidates when sampling with non-zero temperature (default: 5)
    --beam_size BEAM_SIZE
                          number of beams in beam search, only applicable when temperature is zero
                          (default: 5)
    --patience PATIENCE   optional patience value to use in beam decoding, as in
                          https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional
                          beam search (default: None)
    --length_penalty LENGTH_PENALTY
                          optional token length penalty coefficient (alpha) as in
                          https://arxiv.org/abs/1609.08144, uses simple length normalization by default
                          (default: None)
    --suppress_tokens SUPPRESS_TOKENS
                          comma-separated list of token ids to suppress during sampling; '-1' will suppress
                          most special characters except common punctuations (default: -1)
    --initial_prompt INITIAL_PROMPT
                          optional text to provide as a prompt for the first window. (default: None)
    --condition_on_previous_text
                          if True, provide the previous output of the model as a prompt for the next
                          window; disabling may make the text inconsistent across windows, but the model
                          becomes less prone to getting stuck in a failure loop (default: False)
    --fp16                whether to perform inference in fp16; True by default (default: False)
    --temperature_increment_on_fallback TEMPERATURE_INCREMENT_ON_FALLBACK
                          temperature to increase when falling back when the decoding fails to meet either
                          of the thresholds below (default: 0.2)
    --compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD
                          if the gzip compression ratio is higher than this value, treat the decoding as
                          failed (default: 2.4)
    --logprob_threshold LOGPROB_THRESHOLD
                          if the average log probability is lower than this value, treat the decoding as
                          failed (default: -1.0)
    --no_speech_threshold NO_SPEECH_THRESHOLD
                          if the probability of the <|nospeech|> token is higher than this value AND the
                          decoding has failed due to `logprob_threshold`, consider the segment as silence
                          (default: 0.6)
    --threads THREADS     number of threads used by torch for CPU inference; supercedes
                          MKL_NUM_THREADS/OMP_NUM_THREADS (default: 0)
Test example
~~~~~~~~~~~~

Transcribe an English male voice with `tiny` model: ::

  $ whisper-benchmark en-male-1 --model-name tiny
  content_frames : 16297
  dtype : torch.float32
  language : en
  start_time : 1699593688.3675494
  end_time : 1699593693.0126545
  elapsed : 4.6451051235198975
  fps : 3508.4243664330047   <-- You'll mainly put your attention to this value
  device : cuda
  audio_id : en-male-1
  version : 0.0.1
  torch_version : 2.0.1+cu117
  cuda_version : 11.7
  python_version : 3.10.12
  whisper_version : 20231106
  numba_version : 0.58.1
  numpy_version : 1.26.1
  threads : 1

Audio source
------------

The audio files are selected from `Wikimedia Commons <https://commons.wikimedia.org/wiki/Main_Page>`_. Here's the list:

- **en-male-1**: `The Call of South Africa", read by Philip Burgers <https://commons.wikimedia.org/wiki/File:%22The_Call_of_South_Africa%22,_read_by_Philip_Burgers.flac>`_
- **en-male-2**: `Nanotechnology lead reading <https://commons.wikimedia.org/wiki/File:0_nanolead_q10.ogg>`_
- **en-male-3**: `Why There's A Cat Curfew in My House <https://commons.wikimedia.org/wiki/File:12_Why_There%27s_A_Cat_Curfew_in_My_House.oga>`_
- **en-female-1**: `Alessia Cara's voice, from Border Crossings on VOA at Jingle Ball 2016 <https://commons.wikimedia.org/wiki/File:Alessia_Cara%27s_voice,_from_Border_Crossings_on_VOA_at_Jingle_Ball_2016.mp3>`_
- **en-female-2**: `Jabberwocky <https://commons.wikimedia.org/wiki/File:Jabberwocky.ogg>`_
- **en-female-3**: `Joely Richardson on the Albert Memorial <https://commons.wikimedia.org/wiki/File:Joely_Richardson_on_the_Albert_Memorial.ogg>`_

Feel free to contribute by adding more audio, especially for non-english language.

Contribute
----------

This project is created with ❤️ for free by `Cloud Mercato`_ under BSD License. Feel free to contribute by submitting a pull request or an issue.

.. _`Cloud Mercato`: https://www.cloud-mercato.com/
