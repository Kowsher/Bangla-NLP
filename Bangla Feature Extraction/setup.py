import setuptools

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='BnFeatureExtraction',
    packages=["BnFeatureExtraction"],
    include_package_data=True,
    package_data = { "BnFeatureExtraction" : ["*"] },
    version='0.2',
    author="Karigor",
    author_email="",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
            'scipy',
            'gensim',
            'numpy',
            'matplotlib',
            'scikit-learn',
            'glove_python'
      ],
     url="https://github.com/Kowsher/Bangla-NLP/tree/master/Bangla%20Feature%20Extraction",
    license="MIT",
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )