import random
import pandas as pd

adjectives = [
    "beautiful", "enchanting", "mysterious", "serene",
    "vibrant", "ancient", "modern", "bustling", "tranquil", "majestic"
]
nouns = [
    "forest", "city", "ocean", "mountain", "sky",
    "river", "desert", "valley", "meadow", "canyon"
]
verbs = [
    "walking", "exploring", "traveling", "wandering", "observing",
    "studying", "photographing", "painting", "writing about", "researching"
]
places = [
    "Europe", "Asia", "Africa", "North America", "South America",
    "Antarctica", "Australia", "the Arctic", "the Caribbean", "the Middle East"
]


def generate_long_sentences():
    sentences = []
    while len(sentences) < 100:
        sentence = "A {} {} in {} is known for its {}, where people often enjoy {} and {} the {}. This place, with its {} and {}, is a perfect example of {} and {}.".format(
            random.choice(adjectives),
            random.choice(nouns),
            random.choice(places),
            random.choice(nouns),
            random.choice(verbs),
            random.choice(verbs),
            random.choice(nouns),
            random.choice(adjectives),
            random.choice(adjectives),
            random.choice(nouns),
            random.choice(nouns)
        )
        if len(sentence) >= 100:
            sentences.append(sentence)
    return sentences


long_sentences = generate_long_sentences()

data = pd.DataFrame({})
data["text"] = long_sentences
data.to_csv("fake_test_essays.csv", index=False)
