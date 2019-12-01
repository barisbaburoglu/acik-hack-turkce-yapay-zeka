from collections import Counter
from keras.models import model_from_json
import json as JSON
import numpy as np
from keras import backend as KBackend

from keras.optimizers import Adam

from keras.preprocessing.sequence import pad_sequences

#Sözcük Etiketleme Fonksiyonu return: list(etiketli cümle),list(sözcük frekansı),list(etiket frekansı)  param :list(cümle)
def predictPosTag(listSentences):
    KBackend.clear_session()
    MAX_LENGTH = 273

    word2index = JSON.loads(open('media/word2index.json').read())

    tag2index = JSON.loads(open('media/tag2index.json').read())

    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

    train_sentences_X = np.loadtxt('media/train_sentences_X.txt', dtype=int)
    test_sentences_X = np.loadtxt('media/test_sentences_X.txt', dtype=int)
    train_tags_y = np.loadtxt('media/train_tags_y.txt', dtype=int)
    test_tags_y = np.loadtxt('media/test_tags_y.txt', dtype=int)

    # load json and create model
    json_file = open('media/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("media/model.h5")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    test_samples_X = []
    print(listSentences)
    for s in listSentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
        test_samples_X.append(s_int)

    test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
    predictions = loaded_model.predict(test_samples_X)
    predictedTags = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})

    sentenceAndTags = []
    for i in range(len(listSentences)):
        wordAndTags = []
        for j in range(len(listSentences[i])):
            wordAndTags.append([listSentences[i][j], predictedTags[i][j]])
        sentenceAndTags.append(wordAndTags)

    #kelime frekans
    words , tags= sentence2WT(sentenceAndTags)
    countedWords = sorted(Counter(words).items(), key=lambda pair: pair[1], reverse=True)
    #etiket frekans
    countedTags = sorted(Counter(tags).items(), key=lambda pair: pair[1], reverse=True)

    """#kelime/etiket olarak bağlayıp saymak için
    wordsAndTags = []
    for i in range(len(listSentences)):
        for j in range(len(listSentences[i])):
            wordsAndTags.append('/'.join([listSentences[i][j], predictedTags[i][j]]))
    listCounted = []
    for w in wordsAndTags:
        listCounted.append([w, wordsAndTags.count(w)])
    """
    return sentenceAndTags, countedWords , countedTags, 99.26

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])

        token_sequences.append(token_sequence)

    return token_sequences

#Etiketli Kelime dosyasını okur  return: list(cümle) param: Dosya yolu
def readCorpus(pathFile):
    filename = pathFile
    corpus = ''
    with open(filename) as f:
        for line in f:
            corpus += line

    corpus = corpus.lower()
    listCorpus = []
    for sentences in corpus.split('./punc'):
        if not len(sentences) == 0:
            tSen = [nltk.tag.str2tuple(t) for t in sentences.split()]
            listCorpus.append(tSen)

    return listCorpus

#Etiketli Cümlelerden etiket ve kelimeleri ayırma return: list(sözcük), list(etiket) param: list(etiketli cümle)
def sentence2WT(taggedsentences):
    words = []
    tags = []
    for s in taggedsentences:
        for wt in s:
            words.append(wt[0])
            tags.append(wt[1])

    return words, tags

def transpositionSeq(listSentences):
    KBackend.clear_session()
    MAX_LENGTH = 273

    word2index = JSON.loads(open('media/transposition/word2index.json').read())

    tag2index = JSON.loads(open('media/transposition/tag2index.json').read())

    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

    train_sentences_X = np.loadtxt('media/transposition/train_sentences_X.txt', dtype=int)
    test_sentences_X = np.loadtxt('media/transposition/test_sentences_X.txt', dtype=int)
    train_tags_y = np.loadtxt('media/transposition/train_tags_y.txt', dtype=int)
    test_tags_y = np.loadtxt('media/transposition/test_tags_y.txt', dtype=int)

    # load json and create model
    json_file = open('media/transposition/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("media/transposition/model.h5")
    print("Loaded model from disk")

    # model compile
    loaded_model.compile(loss='binary_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    test_samples_X = []
    print(listSentences)
    for s in listSentences:
        str = listToString(s)
        s_int = []
        try:
            s_int.append(word2index[str.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
    test_samples_X.append(s_int)

    test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')

    print("-------------------------")
    print(test_samples_X)
    predictions = loaded_model.predict(test_samples_X)

    predictedTags = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})

    sentenceAndTags = []
    for i in range(len(listSentences)):
        wordAndTags = []
        wordAndTags.append([listSentences[i], predictedTags[i]])
        sentenceAndTags.append(wordAndTags)

    #kelime frekans
    words , tags = sentence2WT(sentenceAndTags)

    return words, tags, 99.26

def listToString(s):

    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))
