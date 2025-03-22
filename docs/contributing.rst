Contribution Guide
=======================

Contributions to ``pycoare`` of all forms are welcome!

Please follow these guidelines for contributing to this project:

* `Fork the pycoare repository into your local GitHub repository <https://github.com/pyCOARE/coare/fork>`_
* Clone your forked repository to your local machine

.. code-block:: bash

    $ git clone <url_to_your_forked_repository>
    $ cd coare

* Install the development requirements using `uv <https://docs.astral.sh/uv/>` and activate the virtual environment

.. code-block:: bash

    $ uv sync --dev
    $ .venv/bin/activate

* Create a new branch for your feature

.. code-block:: bash

    $ git checkout -b my-new-feature

* Install the pre-commit hooks for code formatting

.. code-block:: bash

    $ pre-commit install

* Make your changes and commit them

.. code-block:: bash

    $ git commit -am 'Add some feature'

* Push to the branch

.. code-block:: bash

    $ git push origin my-new-feature

* Create a new Pull Request on GitHub (a good title and description are helpful!)
* A maintainer will review your changes and merge them into the main branch!
