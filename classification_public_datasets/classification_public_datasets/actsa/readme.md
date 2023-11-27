

## ACTSA Dataset


The ACTSA (Annotated Corpus for Telugu Sentiment Analysis) corpus has a collection of Telugu sentences taken from different sources which were then pre-processed and manually annotated by native Telugu speakers using our annotation guidelines. The dataset contains 3 categories and has 5410 total tagged samples. Each sentence has either +1, 0 or -1 label which correspond to positive, neutral and negative.

Originally, presented in this [ACL paper](https://www.aclweb.org/anthology/W17-5408/). Downloaded from [here](https://github.com/NirantK/bharatNLP/releases)


#### Format

The dataset is split into train, valid and test in the ratio 8:1:1, and each part is stored separately in a csv file. Each record in the csv file has the following format: `<label>,<sentence>`
