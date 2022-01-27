Word Embeddings Benchmarks
=====


Dependencies
======

Please see ``requirements.txt``.

Install
======

This package uses setuptools. You can install it running::

    python setup.py install

If you have problems during this installation. First you may need to install the dependencies::

    pip install -r requirements.txt

If you already have the dependencies listed in ``requirements.txt`` installed,
to install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

You can also install it in development mode with::

    python setup.py develop


For Assignment 1
========
* run ``evaluate_similarity.py`` for evaluating all datasets and all embeddings except BERT.

* run ``evaluate_bert.py`` for evaluating all datasets on BERT embedding.

License
=======
Code is licensed under MIT, however available embeddings distributed within package might be under different license. If you are unsure please reach to authors (references are included in docstrings)
