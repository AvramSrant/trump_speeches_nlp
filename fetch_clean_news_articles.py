import nltk
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests as req
import re
from nltk.corpus import stopwords
import calendar as c

# getting the names if the months,days of the week,seasons to remove them
# from the text later on
months=[c.month_name[i].lower() for i in range(1,13)]
days=[c.day_name[i].lower() for i in range(7)]
seasons=['winter','summer','autumn','spring']
time_related=months+days+seasons



urls=['https://www.theguardian.com/us-news/live/2019/jun/28/kamala-harris-democratic-debate-2020-candidates-live-donald-trump-putin',
      'https://www.theguardian.com/us-news/2019/jun/28/memphis-hospital-suing-own-workers-unpaid-medical-bills',
      'https://www.theguardian.com/world/2019/jun/28/german-far-right-group-used-police-data-to-compile-death-list',
      'https://www.theguardian.com/us-news/2019/jun/28/new-york-city-pride-2019-marches-stonewall-50',
      'https://www.theguardian.com/film/2019/jun/28/keanu-reeves-supports-rome-cinema-collective-attacked-by-far-right']



# Stage 1 Getting the data - a newspaper article

def request_article_from_www(url):
    """
    
    Returns text in html format.
    
    Parameter
    ----------
    url : str
    
    """
    
    # Getting the article in HTML from www 
    r = req.get(url)

    # Setting the correct text encoding of the HTML page
    r.encoding = 'utf-8'

    # Extracting the HTML from the request object
    html = r.text

    return html

def request_all_urls():
    """

    Returns a dictionary whose keys are url adresses from the 'urls' list
    and whose values are the texts in html format corresponding to each url.
    
    """
    
    archive={}

    for u,l in zip(urls,list(range(len(urls)))):
        archive['article '+str(l)]=request_article_from_www(u)

    return archive

# Stage 2 Extracting the text from the HTML

def get_text_from_html(archive_rec):
    """

    Returns a list of paragraphs of the article corresponding to given url.

    Parameter
    ----------
    archive_rec : str. All valid archive_rec are the
    values of the dictionary request_all_urls().
    
    """
    
    # Creating a BeautifulSoup object from the HTML
    soup = bs(archive_rec,"html.parser")

    # Extracting the list of paragraphs of the article
    results=soup.find_all('p')

    # Purify it a bit more
    better_results=[]

    for r in results:
        if not (str(r)[:8] in ['<p class','<a class']):
            better_results.append(str(r))

    return better_results

def get_text_from_all_htmls():
    """

    Returns a dictionary whose values are lists of paragraphs of the article
    corresponding to given url.
    
    """
    
    article_base={}
    lengths=list(range(len(request_all_urls().keys())))
    
    for k,leng in zip(request_all_urls().keys(),lengths):
        article_base['beautiful article '+str(leng)]=get_text_from_html(request_all_urls()[k])

    return article_base

def cleanhtml(text):
    """

    Returns a text without html tags.

    Parameter
    ----------
    text : str
    
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    return cleantext

clean_articles_dict = {k: cleanhtml(' '.join(v)) for k, v in get_text_from_all_htmls().items()}


# Stage 3 Tokenization - transforming an article into list of words

def tokenize(clean_article):
    """

    Returns a list of words or dots from the article.

    Parameter
    ----------
    clean_article : str
    
    """
    
    # Creating a tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|\.')

    # Tokenizing the text
    tokens = tokenizer.tokenize(clean_article)

    return tokens


# tokenizing the texts in  clean_articles_dict.values()
# and storing the results in a new dictionary "cleaner_articles_dict"
cleaner_articles_dict={key: tokenize(value) for key, value in clean_articles_dict.items()}


# Identify proper nouns in the text

def identify_proper_nouns(tokens):
    """

    Returns a list of tokens which are the names of people,places,organizations,etc.

    Parameter
    ----------
    tokens : list
    
    """
    
    # getting the list of peoples and places (proper nouns) mentioned in the speech
    persons_things=[]

    for t in tokens:
        if (tokens[(tokens.index(t)-1)]!='.') & ((t.capitalize()==t) | (t.isupper())):
            persons_things.append(t)

    return persons_things

def identify_proper_nouns_for_all_articles():
    """

    Returns a dictionary whose values are lists of tokens which are
    the names of people,places,organizations,etc. (called proper nouns).
    
    """
    
    articles_proper_nouns={}
    length=list(range(len(cleaner_articles_dict.values())))
    
    for value,l in zip(cleaner_articles_dict.values(),length):
        articles_proper_nouns['proper nouns in article '+str(l)]=identify_proper_nouns(value)

    return articles_proper_nouns

proper_nouns=identify_proper_nouns_for_all_articles()

def list_out_all_proper_nouns():
    """

    Returns a list which elements are the proper nouns from all articles.
    
    """
    
    pn_to_remove=[]
    
    for v in proper_nouns.values():
        for i in v:
            pn_to_remove.append(i)

    return pn_to_remove

# I need to compare each value of cleaner_articles_dict (that value is a list of words) with the list pn_to_remove
# I need to preserve only those words that are NOT in the pn_to_remove list
# I need to create a new dictionary with the same keys as cleaner_articles_dict
# but new values - the lists of words shortened by the amount of proper nouns they previously contained

def remove_proper_nouns(dictionary):
    """

    Returns a dictionary which values are the lists of tokens
    without proper nouns corresponding to each article.

    Parameter
    ----------
    dictionary : dictionary
    
    """
    
    no_proper_nouns={}
    span=list(range(len(dictionary.values())))
    
    for list_of_words,l in zip(dictionary.values(),span):
        for word in list_of_words:
            if word in list_out_all_proper_nouns():
                list_of_words.remove(word)
                no_proper_nouns['article no_pn '+str(l)]=list_of_words

    return no_proper_nouns

no_pn_dict=remove_proper_nouns(cleaner_articles_dict)

# getting list of english stopwords,that is,words like am,you etc.
stopwords=stopwords.words('english')

# remove dots
# After we have dealt with (most of) proper nouns we don't need to count dots as words anymore.
# Tokenization - without punctuation

def tokenize_to_remove_punct(no_pn_dict_value):
    """

    Returns a list of tokens without punctuation,stopwords,numbers,calendar names.

    Parameter
    ----------
    no_pn_dict_value : the value of the dictionary no_pn_dict
    
    """
    
    # Creating a tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    # Tokenizing the text
    tokens = tokenizer.tokenize(' '.join(no_pn_dict_value))

    # make all words lowercase
    tokens=[t.lower() for t in tokens]

    # removing stopwords
    tokens=[t for t in tokens if t not in stopwords]

    # removing numbers
    tokens=[t for t in tokens if not t.isdigit()]

    # removing months/days/seasons
    tokens=[t for t in tokens if t not in time_related]

    return tokens

no_pn__no_dots_dict={k:tokenize_to_remove_punct(v) for k,v in no_pn_dict.items()}


# Stage 4 Bringing the words to their base form - lemmatization

def identify_part_of_speech(art_dict_value):
    """

    Returns a list of tuples which elements are the tuple of words
    and it's part of speech and a single integer denoting the position
    of the word in the text.

    Parameter
    ----------
    art_dict_value : the value of the dictionary no_pn__no_dots_dict
    
    """
    
    # To lemmatize correctly, we need to find out which words are noun,
    # which are verbs etc.
    # Clasifying words and saving each word and its respective
    # part of speach into words_and_its_part_of_speech
    words_and_its_part_of_speech=nltk.pos_tag(art_dict_value)

    # saving words together with their location
    words_loc=[(word_loc,length) for word_loc,length in zip(
        words_and_its_part_of_speech,list(range(len(words_and_its_part_of_speech))))]

    return words_loc

# identifying part of speech of each word in dict value list
# and storing the results in a new dict
articles_words_with_pos={key:identify_part_of_speech(value)\
                         for key,value in no_pn__no_dots_dict.items()}

def lemmatize(article_dict_value):
    """

    Returns a list of lemmatized words.

    Parameter
    ----------
    article_dict_value : the value of the dictionary articles_words_with_pos
    
    """
    
    # lemmatizing nouns,verbs and adjectives and storing them to
    # separate lists, together with their locations

    words_loc_noun=[]
    words_loc_verb=[]
    words_loc_adjective=[]

    for pair_of_pairs,length in zip(article_dict_value,list(range(len(article_dict_value)))):
            if ((pair_of_pairs[0][1]=='NN') | (pair_of_pairs[0][1]=='NNS')\
                | (pair_of_pairs[0][1]=='NNP') |(pair_of_pairs[0][1]=='NNPS')):
                
                words_loc_noun.append((nltk.stem.WordNetLemmatizer()\
                                    .lemmatize(pair_of_pairs[0][0],pos='n'),length))
            
            elif ((pair_of_pairs[0][1]=='VB') | (pair_of_pairs[0][1]=='VBN')\
                  | (pair_of_pairs[0][1]=='VBG')|(pair_of_pairs[0][1]=='VBD')| \
                  (pair_of_pairs[0][1]=='VBP') | (pair_of_pairs[0][1]=='VBZ')):
                
                words_loc_verb.append((nltk.stem.WordNetLemmatizer()\
                                    .lemmatize(pair_of_pairs[0][0],pos='v'),length))
                
            elif ((pair_of_pairs[0][1]=='JJ') | (pair_of_pairs[0][1]=='JJR')\
                  | (pair_of_pairs[0][1]=='JJS')):
                words_loc_adjective.append((nltk.stem.WordNetLemmatizer()\
                                         .lemmatize(pair_of_pairs[0][0],pos='a'),length))

            # putting all lemmatized words together
            lemmatized=words_loc_noun+words_loc_verb+words_loc_adjective

    return lemmatized


# performing lemmatization on each article in the dictionary values and storing it in a new dictionary
articles_lemmatized={article:lemmatize(list_of_tuples_of_tuples)\
                     for article,list_of_tuples_of_tuples in articles_words_with_pos.items()}
        
# Putting the words back in the original order after lemmatization
# sorting lemmatized words by their position in the text
articles_lemmatized_sorted={art_lem:sorted(list_of_tuples,key=lambda tup:tup[1])\
                            for art_lem,list_of_tuples in articles_lemmatized.items()}

# we have the words in right order,we no lnger needs adresses
# we will convert to values of the dictionary from list of tuples to list of words

def select_words_from_tuples(list_of_tuples):
    """

    Returns a list of words.

    Parameter
    ----------
    list_of_tuples : the value of the dictionary articles_lemmatized_sorted
    
    """
    
    new_list=[]

    for tup in list_of_tuples:
        new_list.append(tup[0])

    return new_list

lemmatized_words_only={k:select_words_from_tuples(v) for k,v in articles_lemmatized_sorted.items()}

# printing out the first 50 words of each lemmatized article
for key,val in lemmatized_words_only.items():
    print(key,' : ',' '.join(val[:50]))
    
# saving each article to a separtate csv file
for key,value in lemmatized_words_only.items():
    df=pd.DataFrame(lemmatized_words_only[key],columns=[key])
    df.to_csv(r'Documents/{}.csv'.format(key))
