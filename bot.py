#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is a simple echo bot using decorators and webhook with aiohttp
# It echoes any incoming text messages and does not use the polling method.

import sqlite3, random, requests



import re
import random
import glob
import pickle
from functools import reduce
from telebot import types


class MarkovChain:
    def __init__(self):
        self.tree = dict()

    '''
    Trains the generator on a block of text.
    '''
    def train(self, text, factor=1):
        words = filter(lambda s: len(s) > 0, re.split(r'[\s"]', text))
        words = [w.lower() for w in words]
        for a, b in [(words[i], words[i + 1]) for i in range(len(words) - 1)]:
            if a not in self.tree:
                self.tree[a] = dict()
            self.tree[a][b] = factor if b not in self.tree[a] else self.tree[a][b] + self.tree[a][b] * factor

    '''
    Trains the generator on a single file.
    '''
    def train_on_file(self, filename, encodings=None, verbose=False):
        encodings = encodings if encodings is not None else ['utf-8', 'ISO-8859-1']
        ret = False
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as f:
                    self.train(f.read())
                if verbose:
                    print('Successfully trained on "{0}". [ENCODING: {1}]'.format(filename, encoding))
                ret = True
                break
            except UnicodeDecodeError:
                if verbose:
                    print('Unable to decode "{0}" for training. [ENCODING: {1}]'.format(filename, encoding))

        if verbose:
            print()

        return ret

    '''
    Serializes the tree to a file.
    '''
    def save_training(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.tree, f)

    '''
    Deserializes the tree from a file.
    '''
    def load_training(self, file):
        with open(file, 'rb') as f:
            self.tree = pickle.load(f)

    '''
    Trains the generator on a single file, or on a list of files, and saves the state to disk upon finishing. (Uses glob patterns!)
    Returns the number of files successfully parsed and trained on.
    '''
    def bulk_train(self, path, verbose=False):
        i = 0
        for filename in glob.glob(path):
            if self.train_on_file(filename, verbose=verbose):
                i += 1
            elif verbose:
                print('Unable to train on "{0}".'.format(filename))

        if verbose:
            print('Successfully trained on {0} files.'.format(i))

        return i

    '''
    Yields a sequence of words until a dead end is found or until a maximum length, if specified, is reached.
    '''
    def generate(self, start_with=None, max_len=0, rand=lambda x: random.random() * x, verbose=False):
        if len(self.tree) == 0:
            return

        word = start_with if start_with is not None else random.choice([key for key in self.tree])

        if verbose:
            print('Generating a sentence of {0}, starting with "{1}":\n'
                  .format('max. {0} words'.format(max_len) if max_len > 0 else 'unspecified length', word))

        yield word

        i = 1
        while max_len == 0 or i < max_len:
            i += 1
            if word not in self.tree:
                return

            dist = sorted([(w, rand(self.tree[word][w] / len(self.tree[word]))) for w in self.tree[word]],
                          key=lambda k: 1-k[1])
            word = dist[0][0]
            yield word

    '''
    Same as generate(), but formats the output nicely.
    '''
    def generate_formatted(self, word_wrap=80, soft_wrap=True, newline_chars='.?!', capitalize_chars='.?!"',
                           start_with=None, max_len=0, rand=lambda x: random.random() * x, verbose=False):
        ww = 0
        lc = capitalize_chars[0] if len(capitalize_chars) > 0 else ''

        for w in self.generate(start_with=start_with, max_len=max_len, rand=rand, verbose=verbose):
            wstr = w.capitalize() if lc in capitalize_chars else w[0] + w[1:].capitalize() if w[0] in capitalize_chars else w
            wstr += ' ' if w[-1] not in newline_chars else '\n'

            if word_wrap > 0:
                ww += len(wstr)
                if wstr[-1] == '\n':
                    ww = 0

                if ww >= word_wrap:
                    if soft_wrap:
                        wstr += '\n'
                        ww = 0
                    else:
                        i = len(wstr) - ww + word_wrap
                        wstr = wstr[:i] + '\n' + wstr[i:]
                        ww -= word_wrap

            yield wstr
            lc = wstr[-2]

    '''
    Adjusts the relationships between branch and leaf according to a fitness function f.
    '''
    def adjust_weights(self, max_len=2, f=lambda a, b: 0):
        pairs = [w.lower() for w in self.generate(max_len=max_len, rand=lambda r: random.random() * r)]
        pairs = [[pairs[i], None if i == len(pairs) - 1 else pairs[i + 1]] for i in range(len(pairs))][:-1]
        factors = [(f(*p) - 0.5) * 2 for p in pairs]
        for p, x in zip(pairs, factors):
            if x < -1 or x > 1:
                raise ValueError(x)
            self.train(reduce(lambda a, b: '{0} {1}'.format(a, b), p), x)

    '''
    Calls adjust_weights with the multiplied result of multiple fitness functions, for a given number of iterations.
    If verbose==True, shows a neat progress bar.
    '''
    def bulk_adjust_weights(self, fitness_functions=None, iterations=1, pbar_len=14, verbose=False):

        import sys

        if fitness_functions is None or len(fitness_functions) == 0:
            return

        if verbose:
            print('Beginning training with {0} algorithms.'.format(len(fitness_functions)))

        for i in range(iterations):
            self.adjust_weights(f=lambda a, b: reduce(lambda x, y: x * y, [ff(a, b) for ff in fitness_functions]))
            if verbose and i % (iterations // pbar_len + 1) == 0:
                progress = i / iterations
                pbar_full = int(progress * pbar_len)
                pbar_empty = pbar_len - pbar_full

                print('\r[{0}{1}] - {2:.2f}%'.format('=' * pbar_full, '-' * pbar_empty, progress * 100), end='')
                sys.stdout.flush()

        if verbose:
            print('\r[{0}] - {1:.2f}%'.format('=' * pbar_len, 100))
            print('Training complete.')
			
			
import random

#фитнесс функции для Маркова

def aw_none(a, b):
    return 0.5


def aw_random(a, b):
    return random.random()


def aw_favor_simplicity(a, b):
    return len(set([c for c in a + b])) / len(a + b)


def aw_favor_complexity(a, b):
    return 1 - aw_favor_simplicity(a, b)


def aw_favor_alternating_complexity(a, b):
    return (aw_favor_simplicity(b, b) + aw_favor_complexity(a, a)) / 2


def aw_favor_rhymes(a, b):
    a, b = sorted([a, b], key=min)
    return sum([1 if p[0] == p[1] else 0 for p in zip(b[len(b) - len(a):], a)]) / len(a)


def aw_favor_alliterations(a, b):
    a, b = sorted([a, b], key=min)
    return sum([1 if p[0] == p[1] else 0 for p in zip(b[:len(b) - len(a)], a)]) / len(a)


def aw_favor_vowels(a, b):
    return sum([1 if c in 'aeiouy' else 0 for c in a + b]) / len(a + b)


def aw_favor_consonants(a, b):
    return 1 - aw_favor_vowels(a, b)


def aw_favor_punctuation(a, b):
    return sum(map(lambda x: 0.5 if x[-1] in '.,;?!:()[]{}' else 0, [a, b]))


def aw_favor_illegibility(a, b):
    return 1 - aw_favor_punctuation(a, b)



def aw_mul(f, k):
    return lambda a, b: f(a, b) * k


	
	
	
	
	
#Генерируем марковскую цепь	
def gen(m):
    return ''.join([w for w in m.generate_formatted(word_wrap=100, soft_wrap=True, start_with=None, max_len=70, verbose=True)])


mkv = MarkovChain()
mkv.bulk_train('horoscopes-short_(1).txt', verbose=True)

mkv.bulk_adjust_weights(fitness_functions=[aw_mul(aw_favor_complexity, 1/3), aw_mul(aw_favor_punctuation, 3/4),
                                           aw_mul(aw_favor_consonants, 1/2)],
                        iterations=100000,
                        verbose=True)







class User:
    def __init__(self, name):
        self.name = name
        self.age = None
        self.sex = None
		
user_dict = {}





import logging
import ssl

from aiohttp import web

import telebot

API_TOKEN = '1007164978:AAEs7V_KF99TwC_e6T9fSJNuqs-jEEqlz8Q'

WEBHOOK_HOST = '35.226.6.153'
WEBHOOK_PORT = 8443  # 443, 80, 88 or 8443 (port need to be 'open')
WEBHOOK_LISTEN = '0.0.0.0'  # In some VPS you may need to put here the IP addr

WEBHOOK_SSL_CERT = './webhook_cert.pem'  # Path to the ssl certificate
WEBHOOK_SSL_PRIV = './webhook_pkey.pem'  # Path to the ssl private key

# Quick'n'dirty SSL certificate generation:
#
# openssl genrsa -out webhook_pkey.pem 2048
# openssl req -new -x509 -days 3650 -key webhook_pkey.pem -out webhook_cert.pem
#
# When asked for "Common Name (e.g. server FQDN or YOUR name)" you should reply
# with the same value in you put in WEBHOOK_HOST

WEBHOOK_URL_BASE = "https://{}:{}".format(WEBHOOK_HOST, WEBHOOK_PORT)
WEBHOOK_URL_PATH = "/{}/".format(API_TOKEN)

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.TeleBot(API_TOKEN)

app = web.Application()


# Process webhook calls
async def handle(request):
    if request.match_info.get('token') == bot.token:
        request_body_dict = await request.json()
        update = telebot.types.Update.de_json(request_body_dict)
        bot.process_new_updates([update])
        return web.Response()
    else:
        return web.Response(status=403)


app.router.add_post('/{token}/', handle)


# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    msg = bot.reply_to(message, "Привет! Я Бот-Гороскот! А тебя как зовут?")
    bot.register_next_step_handler(msg, process_name_step)


# Handle all other messages
@bot.message_handler(func=lambda message: True, content_types=['text'])
def echo_message(message):
	hor_en = gen(mkv)
	#Перевод на русский отобранного гороскопа
	url = "https://translate.yandex.net/api/v1.5/tr.json/translate?key=trnsl.1.1.20191109T230210Z.7b6ce11b087194c3.18055abead9a7f60d3682f25d16f6774139a7eba&text=" + hor_en + "&lang=en-ru"
	r = requests.get(url = url)
	hor = r.json()['text']
	bot.reply_to(message, hor)
	
GoodHors = [["As far as your makeup is concerned, you must highlight your eyes and lips. Subtle makeup will suit you to a T and reinforce your natural and spontaneous personality, even it takes longer to apply. Silver eyeshadow will allow you to grab the limelight, literally! But be careful not to cover yourself in glitter, it would not have the desired effect.","You cannot stand men who always sit on the fence, and you hate being contradicted once you have made your mind up. You would rather do the things you want, without being pressured into doing anything you’re not that keen on."],["Proud, passionate and determined, you wear you heart on your sleeve and insist on living life to the fullest on a daily basis. Your need to be loved and accepted means that you feel the urge to help, protect, support and advise.","You look very elegant naked and you don’t hesitate to take your clothes off. Extremely lively in the bedroom, you don’t hesitate to take control of the situation under the quilt. Men must be aware that your spine will be the key to awakening all your senses."]]

#ПЕРСОНАЛИЗАЦИЯ!!!
#Ищем расстояние до эталонного гороскопа
#Костыли
def Dist(hors, message):
	user = user_dict[message.chat.id]
	if (user.age < 40):
		GH2=0
		if (user.sex == u'Мужчина'):
			GH1=0
		else:
			GH1=1
	else:
		GH2=1
		if (user.sex == u'Мужчина'):
			GH1=0
		else:
			GH1=1
	etol_hor = GoodHors[GH1][GH2]
	
	corpus = hors.append(etol_hor)
	vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
	tfidf = vect.fit_transform(corpus)                                                                                                                                                                                                                       
	pairwise_similarity = tfidf * tfidf.T
	column =  [x[-1] for x in pairwise_similarity]
	max_similarity = max(column[0:-1])
	i = column.index(max_similarity)
	return corpus[i]
	
def process_name_step(message):
    try:
        chat_id = message.chat.id
        name = message.text
        user = User(name)
        user_dict[chat_id] = user
        msg = bot.reply_to(message, 'Сколько Вам лет?')
        bot.register_next_step_handler(msg, process_age_step)
    except Exception as e:
        bot.reply_to(message, 'oooops ' + ' process_name_step')


def process_age_step(message):
    try:
        chat_id = message.chat.id
        age = message.text
        if not age.isdigit():
            msg = bot.reply_to(message, 'Введите целое число!')
            bot.register_next_step_handler(msg, process_age_step)
            return
        user = user_dict[chat_id]
        user.age = age
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        markup.add('Мужчина', 'Женщина')
        msg = bot.reply_to(message, 'Ваш пол?', reply_markup=markup)
        bot.register_next_step_handler(msg, process_sex_step)
    except Exception as e:
        bot.reply_to(message, 'oooops ' + ' process_age_step')


def process_sex_step(message):
    try:
        chat_id = message.chat.id
        sex = message.text
        user = user_dict[chat_id]
        if (sex == u'Мужчина') or (sex == u'Женщина'):
            user.sex = sex
        else:
            raise Exception()
        bot.send_message(chat_id, 'Приятно познакомится, ' + user.name + '\n Возраст:' + str(user.age) + '\n Пол:' + user.sex + '\n Если хотите персональный гороскоп, то отправьте мне что-нибудь') 
    except Exception as e:
        bot.reply_to(message, 'oooops ' + ' process_sex_step')


bot.enable_save_next_step_handlers(delay=2)


bot.load_next_step_handlers()



bot.remove_webhook()


bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH,
                certificate=open(WEBHOOK_SSL_CERT, 'r'))


context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain(WEBHOOK_SSL_CERT, WEBHOOK_SSL_PRIV)


web.run_app(
    app,
    host=WEBHOOK_LISTEN,
    port=WEBHOOK_PORT,
    ssl_context=context,
)



