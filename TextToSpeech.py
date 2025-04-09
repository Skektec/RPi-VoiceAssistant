from realtimetts import TextToAudioStream, CoquiEngine

engine = CoquiEngine()
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()