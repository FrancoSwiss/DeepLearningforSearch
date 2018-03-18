from whoosh.fields import Schema, TEXT, NGRAMWORDS
from whoosh.analysis import CharsetFilter, StemmingAnalyzer
from whoosh.support.charset import accent_map
from whoosh import index
import os.path
import glob
import codecs
import shutil
import warnings
# Remove annoying warning from gensim
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

# Source files path
TEXT_PATH = 'corpus'

# Index path
INDEX_PATH = 'corpus_index'

# Output Word2Vec files
OUTPUT_W2VMODEL_BIN = 'w2v_model.bin'
OUTPUT_W2VMODEL_VEC = 'w2v_model.vec'

# Length of short sentences to eliminate during Word2Vec training
SENTENCE_LIMIT = 30

# Word2Vec model dimension
DIMENSION = 100

# Limit whoosh index size (good for the debug)?
INDEX_LIMITED = False

# Size of whoosh index limit
INDEX_SIZE = 500

# Prepare Word2Vec model?
TRAIN_WORD2VEC = True

# Use NGram?
USE_NGRAM = True

# NGram Minimum size
NGRAM_MIN_SIZE = 2

# NGram Minimum size
NGRAM_MAX_SIZE = 4


def prepare_text(txt):
    """
    Simple tokenization with extraction of stop words and non-letter symbols
    """
    doc = WordPunctTokenizer().tokenize(txt)
    doc = [word for word in doc if word not in stopword_set]
    doc = [word for word in doc if word.isalpha()]
    return doc


stopword_set = set(stopwords.words('german'))

# Index schema
analyzer = StemmingAnalyzer() | CharsetFilter(accent_map)
if USE_NGRAM:
    schema = Schema(title=TEXT(stored=True),
                    body=TEXT(analyzer=analyzer, stored=True),
                    ngrams=NGRAMWORDS(minsize=NGRAM_MIN_SIZE, maxsize=NGRAM_MAX_SIZE, stored=False, at=None))
else:
    schema = Schema(title=TEXT(stored=True),
                    body=TEXT(analyzer=analyzer, stored=True))

# Empty index folder if needed
if os.path.exists(INDEX_PATH):
    shutil.rmtree(INDEX_PATH, ignore_errors=True)
os.mkdir(INDEX_PATH)

# Remove possible Word2Vec remains
for f in glob.glob(OUTPUT_W2VMODEL_BIN+'.*'):
    os.remove(f)

# Get list of files to process
files = glob.glob(os.path.join(TEXT_PATH, '*.txt'))

if INDEX_LIMITED:
    files = files[:INDEX_SIZE]

# Populate index and Word2Vec model
train_data = []
ix = index.create_in(INDEX_PATH, schema)
writer = ix.writer()
num_files = len(files)
for i, item in enumerate(files):
    print('Processing file: %i from %i (%.2f%%).' % (i+1, num_files, (i+1)/num_files*100))
    text = codecs.open(item, 'r', 'utf-8').read()
    if USE_NGRAM:
        writer.add_document(title=os.path.splitext(os.path.basename(item))[0], body=text, ngrams=text)
    else:
        writer.add_document(title=os.path.splitext(os.path.basename(item))[0], body=text)
    if TRAIN_WORD2VEC:
        sent = sent_tokenize(text)
        for s in sent:
            s = s.strip()
            # Eliminate short sentences in the training data
            if len(s) < SENTENCE_LIMIT:
                continue
            train_data.append(prepare_text(s.lower()))

print('Committing whoosh index changes...')
writer.commit()
if TRAIN_WORD2VEC:
    print('Preparing Word2Vec model...')
    model = gensim.models.Word2Vec(train_data, size=DIMENSION, sg=0)
    model.save(OUTPUT_W2VMODEL_BIN)
    model.wv.save_word2vec_format(OUTPUT_W2VMODEL_VEC)
print('Done.')
