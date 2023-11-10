Whisper Benchmark
========================

**Whisper-Benchmark** is a simple tool to evaluate performance of whisper.

Install
-------

::

  pip install https://github.com/cloudmercato/whisper-benchmark/archive/refs/heads/master.zip
  
  
Usage
-----

Command line
~~~~~~~~~~~~

Test example: File uploading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  $ whisper-benchmark en-male-1 --model-name tiny
  content_frames : 16297
  dtype : torch.float32
  language : en
  start_time : 1699593688.3675494
  end_time : 1699593693.0126545
  elapsed : 4.6451051235198975
  fps : 3508.4243664330047
  device : cuda


Contribute
----------

This project is created with ❤️ for free by `Cloud Mercato`_ under BSD License. Feel free to contribute by submitting a pull request or an issue.

.. _`Cloud Mercato`: https://www.cloud-mercato.com/
