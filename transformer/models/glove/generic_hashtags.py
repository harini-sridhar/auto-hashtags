
import pickle

generic_hashtags = ['love',
'instagood',
'cute',
'photooftheday',
'instamood',
'tweegram',
'picoftheday',
'igers',
'instadaily',
'instagramhub',
'follow',
'igdaily',
'bestoftheday',
'instagramers',
'like',
'life',
'photo',
'instahub',
'followme',
'like4like',
'instalike',
'likeforlike',
'photography',
'l4l',
'instapic',
'instacool',
'instalove',
'webstagram',
'picture',
'instaphoto'
]

with open('generic_hashtags.pkl', 'wb') as f:
    pickle.dump(generic_hashtags, f)
