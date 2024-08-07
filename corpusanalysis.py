import re
from collections import Counter
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import nltk
from nltk.probability import FreqDist
nltk.download('punkt')
#Corpus Analysis for Hindi Language
def normalize_text(text, lang='hi'):
    normalizer_factory = IndicNormalizerFactory()
    normalizer = normalizer_factory.get_normalizer(lang)
    normalized_text = normalizer.normalize(text)
    return normalized_text

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def type_token_analysis(text):
    tokens = indic_tokenize.trivial_tokenize(text)
    types = set(tokens)
    type_token_ratio = len(types) / len(tokens)
    return type_token_ratio, tokens, types

def syllable_analysis(text):
    syllables = []
    words = indic_tokenize.trivial_tokenize(text)
    
    for word in words:
        syllables.extend(re.findall(r'[क-ह]+', word))  # Syllable extraction regex for Hindi Devanagari script
    
    syllable_freq = FreqDist(syllables)
    most_common_syllables = syllable_freq.most_common(10)
    
    initial_syllables = Counter()
    medial_syllables = Counter()
    final_syllables = Counter()
    
    for word in words:
        word_syllables = re.findall(r'[क-ह]+', word)
        if word_syllables:
            initial_syllables[word_syllables[0]] += 1
            if len(word_syllables) > 1:
                medial_syllables.update(word_syllables[1:-1])
            if len(word_syllables) > 1:
                final_syllables[word_syllables[-1]] += 1
    
    return most_common_syllables, initial_syllables.most_common(10), medial_syllables.most_common(10), final_syllables.most_common(10)

def main():
    corpus_file_path = 'demotext.txt'  
    text = load_corpus(corpus_file_path)
    text = normalize_text(text)

    # Type-Token Analysis using Frequency
    type_token_ratio, tokens, types = type_token_analysis(text)
    print(f'Type-Token Ratio: {type_token_ratio}')
    
    # Syllable Analysis for the given Corpus of Hindi Text
    most_common_syllables, initial_syllables, medial_syllables, final_syllables = syllable_analysis(text)
    
    print(f'Most Frequent Syllables:\n{most_common_syllables}\n')
    print(f'Most Frequent Initial Syllables:\n{initial_syllables}\n')
    print(f'Most Frequent Medial Syllables:\n{medial_syllables}\n')
    print(f'Most Frequent Final Syllables:\n{final_syllables}\n')

if __name__ == '__main__':
    main()

# https://metatext.io/datasets-list/hindi-language
# Hindi Corpus data for kinds of Analysis