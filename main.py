from sd import *

sd = SD()

inputs = [("Brad pitt", 0.1), ("Angelina Jolie", 0.9)]

image = sd.generateFromWeightedTextEmbeddings(inputs=inputs, neg_prompt="", steps=1, cfg=0, seed=None)

image.save("output-1.png")
