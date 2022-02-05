<!DOCTYPE html>
<html>
<body>

<!DOCTYPE html>
<html>
<body>
  
## HolC - (OSNEM) 
Code repo for paper titled 'Balancing between Holistic and Cumulative Sentiment Classification' published in Online Social Networks and Media - Journal 

  ## Contributors

- [Pantelis Agathangelou](https://github.com/ailabunic-panagath)
- [Ioannis Katakis](https://github.com/iokat)

  ## Reference
When using the code-base please use the following reference to cite our work:<br/>
[to be placed when issued]. DOI:


  ## How to run the model
1: The code-base is set to run without additional path settings, if it is downloaded and placed at the downloads folder <br/>
2: The data folder must contain the datasets in excel format. The columns must be arranged in the folowing format:<br/>
   - 1st column : user's opinions,<br/>
   - 3rd column : labels for three classes classification task i.e.:[0,1,2] (for binary simply set n_classes=2 at the 'configs.py' file,<br/>
   - 5th column : labels for five classes or fine-grained classification i.e.:[0,1,2,3,4], if exist,<br/>
   - 6th column : labels for six classes, if exist,<br/>
   - for many classes text classification, simply place the labels at the corresponding column.<br/>
   
3: The "configs.py" file, includes the hyperparameters for training the model.<br/>
4: The embeddings folder must contain the <a href="https://fasttext.cc/docs/en/english-vectors.html">crawl-300d-2M-subword.zip</a> <br/>
5: The train_holc.py is the main file to load & train the model.

  ## Requirements
  <ul>
  <li>tensorflow version: '1.12.0' </li>
  <li>numpy version: '1.16.1'</li>
  <li>sklearn version: '0.19.0'</li>
  <li>nltk version: '3.2.4'</li>
  

    ## License
The framework is open-sourced under the Apache 2.0 License base. The codebase of the framework is maintained by the authors for academic research and is therefore provided "as is".
  
  
 
 
 </div>

</body>
</html>

</body>
</html>

