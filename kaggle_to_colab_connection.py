# -*- coding: utf-8 -*-
"""Kaggle to colab connection

Here we have used NumtaDB ,one of the largest bengali handwritten digit dataset, for the recognition purposes.Besides,we have work on google colab for faster operation instead of jupyter notebook.
In order to normalize the dataset usages ,there's a method to connect the kaggle dataset with google colab.

First of all create a API token from users kaggle Account and download it.Then pass the below codes-
"""


from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json



!kaggle datasets download -d BengaliAI/numta

from zipfile import ZipFile
file_name = "numta.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('Done')
