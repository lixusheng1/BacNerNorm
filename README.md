#  Bacterial named entity recognition and normalization

the python package:
tensowflow
nltk
spacy
scispacy

# How to train a model
1. You should get a word embedding file ,such as word2vec,glove,fastext . And download this file to the /embedding

2. python build_data.py

3. python train.py

4. You can evaluate the model 
    
  python evaluate.py

5. You can predict with the trained model

   python predict.py



# Using the trained model for the large text.

1. bacterial named entity recognition

   python BacNer.py
   
2. bacterial named normalization

   python BacNorm.py
   

