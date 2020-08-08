import pandas as pd
import re
import nltk
import string
import utility

def cleanText(text):
    stripped = text.strip()
    PATTERN = r'[,|.|?|$|#|!|&|*|%|@|(|)|~|^0-9]'
    result = re.sub(PATTERN, r'', stripped)
    return result.lower().replace('"', '') 

def tokenize(text):
    sentences = nltk.sent_tokenize(text)
    wordTokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return wordTokens

def cleanAftertokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filteredTokens =  filter(None, [pattern.sub('', token) for token in tokens])
    return [token for token in tokens if pattern.sub('', token) != None]

def writeCSV(fileName, list):
    with open(fileName, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(list)

def isWordEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def checkGetWordByLanguage(word,isEnglish):
    if (isEnglish and isWordEnglish(word) or ((not isEnglish) and (not isWordEnglish(word)))):
        return word
    else :
        return None

def createLanguageSeparatedList(list,isEnglish):
    # select only sinhala words
    return [text for text in list if checkGetWordByLanguage(text, isEnglish) != None]

def result(outputResult = False):
    data = utility.loadCSVData('Sinhala_Singlish_Hate_Speech.csv')
    data["CleanedPhrase"] = data["Phrase"].apply(cleanText)
    data["TokenFullList"] = data["CleanedPhrase"].apply(tokenize)
    data["SingleList"] = [fullsentence for line in data["TokenFullList"] for fullsentence in line]
    data["Tokens"] = data["SingleList"].apply(cleanAftertokenization)
    data["SinhalaTokens"] = [createLanguageSeparatedList(tokens, False) for tokens in data["Tokens"] ]
    data["EnglishTokens"] = [createLanguageSeparatedList(tokens, True) for tokens in data["Tokens"] ]
    # calculate the percentages
    data["SinhalaTokenPercentage"] = data.apply(lambda row: (len(row.SinhalaTokens) / len(row.Tokens)) * 100, axis=1)
    data["EnglishTokenPercentage"] = data.apply(lambda row: (len(row.EnglishTokens) / len(row.Tokens)) * 100, axis=1)
    
    # print as csv
    if(outputResult):
        utility.writeDFCSV("part1", "result.csv", data)
        utility.writeCSV("part1", "sinhala-word-list.csv", data["SinhalaTokens"].tolist())
        utility.writeCSV("part1", "english-word-list.csv", data["EnglishTokens"].tolist())
    return data
