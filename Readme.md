#### Tested LucasKanade  and LukasKanade with pyramidal extension tracking algorithms on different datasets


Usage:
    
    `python lucas_kanade.py --dataset dataset  --roi x,y,width,height --method method_name`
    
   
LucasKanade worked good with dog dataset, it lost only a small part of tracking points in the end of the video.
LucasKanade with pyramidal extension improved this result, it didn't lose any point at all on dog dataset.

On CarScale dataset LucasKanade algorithm worked well, untill car was passing three.
After three passing only two good points were left.
Pyramidal extension made situation even worse, car lost all points after passing three.

Both methods worked good with mountain bike, no significant difference there.