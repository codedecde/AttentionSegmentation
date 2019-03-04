import nltk
PUNCT_SET = set([".", ",", "!", "-", "?", "'", ")", "("])
# STOPWORDS = {}
STOPWORDS_EN = set(nltk.corpus.stopwords.words("english"))
STOPWORDS_ES = set(nltk.corpus.stopwords.words("spanish"))
STOPWORDS_NL = set(nltk.corpus.stopwords.words("dutch"))

SYM_STOPWORDS_EN = STOPWORDS_EN | PUNCT_SET
SYM_STOPWORDS_ES = STOPWORDS_ES | PUNCT_SET
SYM_STOPWORDS_NL = STOPWORDS_NL | PUNCT_SET