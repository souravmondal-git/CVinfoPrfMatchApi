from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
import mammoth
import pandas as pd
import textract
import io
import os
import nltk
import numpy as np
from nltk import word_tokenize, sent_tokenize
import re
#import spacy
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from CVinfoProfMatch import settings


stop_words = stopwords.words('english')
#nlp = spacy.load('en_core_web_sm')


def extract_pdf(path):
    resource_manager = PDFResourceManager()
    output = io.StringIO()
    text_converter = TextConverter(resource_manager, output, laparams=LAParams())
    interpreted_page = PDFPageInterpreter(resource_manager, text_converter)
    with open(path, 'rb') as file:
        for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
            interpreted_page.process_page(page)
            text = output.getvalue()
    text_converter.close()
    output.close()

    return text


def read_file(filename):
    if (filename.endswith(".pdf")):
        try:
            fileTEXT = extract_pdf(filename)
        except Exception:
            print('Error raised reading the pdf file:' + filename)

    elif (filename.endswith(".docx")):
        try:
            with open(filename, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                fileTEXT = result.value
        except IOError:
            print('Error raised reading the docx file:' + filename)

    elif (filename.endswith(".doc")):
        try:
            fileTEXT = textract.process(filename).decode('utf-8')
        except Exception:
            print('Error raised reading the doc file:' + filename)

    elif (filename.endswith(".txt")):
        try:
            text_file = open(filename, "rt")
            fileTEXT = text_file.read()
        except Exception:
            print('Error raised reading the txt file:' + filename)
    else:
        print(filename + 'not supported!')

    return fileTEXT


def email_ids(text):
    pattern_email = re.compile(r'[A-Za-z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    email_objs = pattern_email.findall(str(text))
    email_objs = list(set(email_objs))
    return email_objs


def phone_number(text):
    pattern = re.compile(
        r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')

    pt = pattern.findall(text)
    pt = [re.sub(r'[,.]', '', ah) for ah in pt if len(re.sub(r'[()\-.,\s+]', '', ah)) > 9]
    pt = [re.sub(r'\D$', '', ah).strip() for ah in pt]
    pt = [ah for ah in pt if len(re.sub(r'\D', '', ah)) <= 15]
    for ah in list(pt):

        if len(ah.split('-')) > 3: continue
        for x in ah.split("-"):
            try:
                if x.strip()[-4:].isdigit():
                    if int(x.strip()[-4:]) in range(1900, 2100):
                        pt.remove(ah)
            except:
                pass
    number = None
    number = list(set(pt))
    return number


def text_processing(x):
    proc_text = []
    for txt in sent_tokenize(x):
        txt = re.sub('\n', ' ', txt)
        txt = re.sub('\s+', ' ', txt)
        txt = re.sub('[:,!]', '', txt)
        txt = txt.replace('\xad', '')
        proc_text.append(txt)

    return " ".join(proc_text)

def remove_emails_links(text):
    sentences = []
    for s in sent_tokenize(text):
        s = re.sub('[A-Za-z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+','',s)
        s = re.sub('[!"`#%&,:;<>=@{}~\$\(\)\*\+\/\\\?\[\]\^\|,a-z,\d+]+.com','',s)
        s = re.sub('[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)','',s)
        s = re.sub('www.','',s)
        s = re.sub('[\d\d\/]+\d\d\d\d','',s)
        sentences.append(s)
    return " ".join(sentences)

#def find_name(text):
#    nlp = spacy.load('en_core_web_sm')
#    text_nlp = nlp(text)
#    ner_tagged = [(word.text, word.ent_type_) for word in text_nlp]
#    names = [i[0] for i in ner_tagged if (i[1]=='PERSON') and (len(i[0])>1)]
#    if (len(names[1]) <= 2):
#        name = " ".join(names[:3])
#    else:
#        name = " ".join(names[:2])
#    return name

def preprocess_cv_sim(text):
    sentences = []
#    nlp_fs_cv = nlp(text)
#    names_fs_cv = [word.text for word in nlp_fs_cv if (word.ent_type_ == 'PERSON')]
    for s in sent_tokenize(text):
        s = re.sub('\n',' ',s)
        s = re.sub('\\n',' ',s)
        s = re.sub('\s+',' ', s)
        s = re.sub('•', '',s)
        s = re.sub('\|','',s)
        s = re.sub('[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '',s)
        s = re.sub('[A-Za-z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+','',s)
        s = re.sub('[!"`#%&,:;<>=@{}~\$\(\)\*\+\/\\\?\[\]\^\|,a-z,\d+]+.com','',s)
        s = re.sub('https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)','',s)
        s = re.sub('[\(\)]','',s)
        s = re.sub('www.','',s)
        s = re.sub('[:,!]','', s)
        s = s.replace('\xad','')
        words = [w.lower() for w in word_tokenize(s) if (w not in stop_words)]
        sentences.append(words)
    return sentences

def cos_similarity(x, y):
    cos = np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))
    return cos

def find_avg_similarity(x, y):
    avg_sim = 0.0
    for i in x:
        for j in y:
            avg_sim += cos_similarity(i,j)
    avg_sim /= (len(x)*len(y))
    return avg_sim


def cv_profiling(filename):
    text = read_file(filename)
    text = text_processing(text)
    email = ",".join(email_ids(text))
    phone = ",".join(phone_number(text))
    filtered_text = remove_emails_links(text)
    #name = find_name(filtered_text)
    model_filepath = os.path.join(settings.MODELS, 'doc2vec_fstack_jd_model.mod')
    jdvec_filepath = os.path.join(settings.MODELS, 'jd_fs_vector.npy')
    model = Doc2Vec.load(model_filepath)
    jd_fs_vector = np.load(jdvec_filepath, allow_pickle=True)

    cv_doc = preprocess_cv_sim(text)
    cv_vector = [model.infer_vector(x) for x in cv_doc]

    prof_match_score = find_avg_similarity(jd_fs_vector, cv_vector)

    return {'Email-ids': email, 'Phone Numbers': phone, 'Profile Matching Score': prof_match_score}