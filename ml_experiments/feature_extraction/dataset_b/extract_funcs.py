import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


def get_ngrams(train, test, min_gram, max_gram, min_df, max_features, analyzer, colname):
    """transforms data into n-gram embeddings"""

    vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=[min_gram, max_gram], lowercase=True, min_df=min_df, max_features=max_features)

    # fit vecotrizer and transform training data
    vector_train = vectorizer.fit_transform(train)
    print(vector_train.shape)

    # get ngram vocabulary
    ngram_names = vectorizer.get_feature_names()

    # add feature specific beginning to column name
    ngram_names = [colname + x for x in ngram_names]
    print(ngram_names)

    # create training data df with respective colnames
    train_emb = pd.DataFrame(vector_train.toarray(), columns=ngram_names)

    # transform test data
    vector_test = vectorizer.transform(test)

    # create test data df with respective colnames
    test_emb = pd.DataFrame(vector_test.toarray(), columns=ngram_names)

    return train_emb, test_emb


def base_features(s):
    numbers = sum(map(str.isdigit, s))
    letters = sum(map(str.isalpha, s))
    spaces = sum(map(str.isspace, s))
    others = len(s) - numbers - letters - spaces
    length = len(s)
    number_perc = numbers/length
    letter_perc = letters/length
    return numbers, letters, spaces, others, length, number_perc, letter_perc


def get_tfidf(train_text, test_text, max_df, max_features, min_df, colname):
    # Compute embedding for both lists
    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english', max_df=max_df,
                                 max_features=max_features, min_df=min_df)

    # fit vecotrizer and transform training data
    vector_train = vectorizer.fit_transform(train_text)
    print(vector_train.shape)

    # get tfidf vocabulary
    tfidf_names = vectorizer.get_feature_names()

    # add feature specific beginning to column name
    tfidf_names = [colname + x for x in tfidf_names]

    # create training data df with respective colnames
    train_emb = pd.DataFrame(vector_train.toarray(), columns=tfidf_names)

    # transform test data
    vector_test = vectorizer.transform(test_text)

    # create test data df with respective colnames
    test_emb = pd.DataFrame(vector_test.toarray(), columns=tfidf_names)

    return train_emb, test_emb


def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    print(imp[::-1][0:top])
    print(names[::-1][0:top])

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()



def f_importances2(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
