
https://codetrips.com/2016/09/19/how-to-publish-a-pyton-package-on-pypi/

to deploy:

.. code:: shell

    pip install twine # for installation
    python setup.py sdist # for building a zip file
    twine upload dist/* # for uploading the zip file
