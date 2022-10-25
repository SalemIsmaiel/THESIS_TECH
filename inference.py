import asyncio

from predictor import Predictor
from preprocessors.microsoftstore import MicrosoftStorePreprocessor

text = """
TikTok is THE destination for mobile videos. On TikTok, short-form videos are exciting, spontaneous, and genuine. Whether you’re a sports fanatic, a pet enthusiast, or just looking for a laugh, there’s something for everyone on TikTok. All you have to do is watch, engage with what you like, skip what you don’t, and you’ll find an endless stream of short videos that feel personalized just for you. From your morning coffee to your afternoon errands, TikTok has the videos that are guaranteed to make your day.

We make it easy for you to discover and create your own original videos by providing easy-to-use tools to view and capture your daily moments. Take your videos to the next level with special effects, filters, music, and more. 

■ Watch endless amount of videos customized specifically for you
A personalized video feed based on what you watch, like, and share. TikTok offers you real, interesting, and fun videos that will make your day.
 
■ Explore videos, just one scroll away
Watch all types of videos, from Comedy, Gaming, DIY, Food, Sports, Memes, and Pets, to Oddly Satisfying, ASMR, and everything in between.
 
■ Pause recording multiple times in one video
Pause and resume your video with just a tap. Shoot as many times as you need.
 
■ Be entertained and inspired by a global community of creators
Millions of creators are on TikTok showcasing their incredible skills and everyday life. Let yourself be inspired.

■ Add your favorite music or sound to your videos for free
Easily edit your videos with millions of free music clips and sounds. We curate music and sound playlists for you with the hottest tracks in every genre, including Hip Hop, Edm, Pop, Rock, Rap, and Country, and the most viral original sounds.

■ Express yourself with creative effects
Unlock tons of filters, effects, and AR objects to take your videos to the next level.

■ Edit your own videos 
Our integrated editing tools allow you to easily trim, cut, merge and duplicate video clips without leaving the app.

* Any feedback? Contact us at https://www.tiktok.com/legal/report/feedback or tweet us @tiktok_us
"""


async def infer_microsoft_store():
    preprocessor = MicrosoftStorePreprocessor()

    dataset = await preprocessor.read_dataset("dataset/Clean-ContextualData22Values.csv")

    predictor = Predictor("models/microsoftstore")
    training_set = dataset["description"]
    predictor.create_corpus(training_set)
    print(len(predictor.corpus))
    print(len(predictor.dictionary.items()))
    print(predictor.infer(text))

if __name__ == '__main__':
    asyncio.run(infer_microsoft_store())
