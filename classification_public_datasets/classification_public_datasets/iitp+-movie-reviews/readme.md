

## IIT-Patna+ Movie Reviews Dataset


The dataset contains movie reviews tagged to one of the three classes - positive, negative or neutral. `conflict` class has been ignored. 

Presented in the paper:
```
Md Shad Akhtar, Ayush Kumar, Asif Ekbal, Pushpak Bhattacharyya; A Hybrid Deep Learning Architecture for Sentiment Analysis; In proceedings of the 26th International Conference on Computational Linguistics (COLING 2016); 482-493; Osaka, Japan; 2016. 
```

Obtained from http://www.iitp.ac.in/~ai-nlp-ml/resources.html

In addition, datasets from [IIT Bombay](http://www.cfilt.iitb.ac.in/resources/senti/HPLC_downloaderInfo.php) and [iNLTK](https://www.kaggle.com/disisbig/hindi-movie-reviews-dataset) project have also been merged. 

#### Format

The dataset is split into train, valid and test in the ratio 8:1:1, and each part is stored separately in a csv file. Each record in the csv file has the following format: `<label>,<review>`
