from setuptools import setup
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'BanglaSpeechRecognition',         
  packages=['BanglaSpeechRecognition'], 
  version = '1.00',    
  license='MIT', 
  description = 'Speech Recognition for Bangla Language developed by Md. Kowsher',   
  long_description=long_description,
  author = 'Md. Kowsher',                   
  author_email = 'ga.kowsher@gmail.com',      
  url = 'https://github.com/Kowsher/Bangla-Speech-Recognition.git', 
  
  install_requires=[  
          'speechrecognition',
       
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
  ],

)