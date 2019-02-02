# Auto-Hashtags

Automatically adding relevant hashtags to images.  
  
## Problem
When using social networks, especially for social media influencers or businesses looking for social media engagement, it would be great if relevant hashtags are automatically generated when uploading an image. 

![Alt text](./images_for_readme/insight_example.jpg?raw=true "Title")

## Pipeline 

![Alt text](./images_for_readme/pipeline.png?raw=true "Title")

## Data

The Microsoft COCO database will be used. Here are examples of the data:

![Alt text](./images_for_readme/coco_eg.jpg?raw=true "Title")

## Model 

A transformer model is used to generate captions for the image.  

The caption is then filtered using the nltk stopwords list, and punctuations/single-characters will be removed to create hashtag-like words for the image.  

These words are embedding using GloVe and are matched to categories in the Instagram popular hashtag database.  

## Results 

![Alt text](./images_for_readme/results.png?raw=true "Title")

## Instructions to Run 

TBD.

