import re


def delete_links(string):
    """deletes link from string"""
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    if len(urls) > 0:
        for url in urls:
            string = string.replace(url, '')
    return string


def remove_twitter_pic_link(s):
    """removes the twitter image link from tweet"""
    ind = 0
    while ind != -1:
        ind = s.find('pic.twitter.com/')
        s = s[:ind] + s[ind+27:]
    return s


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"www\.\S+", "", sample)


def cleanhtml(raw_html):
    """remove html tags"""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def clean_string2(s):
    """cleans string"""
    s = s.replace('&quot;', '')
    s = s.replace('\n', ' ')
    s = s.replace('\xa0…', '')
    s = s.replace('\xa0', '')
    s = s.replace('#', '')
    s = s.replace('@', '')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace(',', '')
    s = s.replace(' ,', ' ')
    s = s.replace(', ', ' ')
    s = s.replace('      ', ' ')
    s = s.replace('     ', ' ')
    s = s.replace('    ', ' ')
    s = s.replace('   ', ' ')
    s = s.replace('  ', ' ')
    return s


def remove_emoji(string):
    """remove emojis, taken from: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def tolower(s):
    return s.lower()


def remove_emoji(string):
    """remove emojis, taken from: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def clean_posts(text):
    text = [delete_links(x) for x in text]
    text = [remove_twitter_pic_link(x) for x in text]
    text = [remove_URL(x) for x in text]
    text = [cleanhtml(x) for x in text]
    text = [tolower(x) for x in text]
    text = [clean_string2(x) for x in text]
    return text