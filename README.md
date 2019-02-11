# Auto-Hashtags
#### "*Key* words to *unlock* your engagement potential!" 

Hashtags play a very important role when businesses and influencers on social media look to increasing their audience engagement. But creating/choosing relevant and popular hashtags is time-consuming and at times difficult too. I have automated this process.
  

## Pipeline 

![Alt text](./images_for_readme/pipeline.png?raw=true "Title")

## Data

The Microsoft COCO database will be used. Here are examples of the data:

![Alt text](./images_for_readme/coco_eg.jpg?raw=true "Title")

## Model 

A transformer model is used to generate captions for the image.  

The caption is then filtered using the nltk stopwords list, and punctuations/single-characters will be removed to create hashtag-like words for the image.  

These words are embedding using GloVe and are matched to categories in the Instagram popular hashtag database.  

## Demo 

![Alt text](./images_for_readme/demo.png?raw=true "Title")

## Instructions to Run 

TBD.

## Credits 

Transformer model: https://github.com/ruotianluo/Transformer_Captioning.

