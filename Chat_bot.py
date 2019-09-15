from config import IEXTOKEN
from iexfinance.stocks import Stock
from iexfinance.refdata import get_symbols
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import random
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
import telebot
from config import TOKEN
from telebot import apihelper







def erase_all():
    global params_stock, params_description, params_append, state, start, wait_for_affirm, affirmative,stockk
    params_description.clear()
    params_stock.clear()
    state = 0
    start = 0
    wait_for_affirm = 0
    affirmative = 0
    stockk.clear()


def interpret(msg):
    data = interpreter.parse(msg)
    return data


def find_stocks(param):
    results = []
    for j in list(symbol_map.keys()):
        if param.upper() == j.upper():
            results.append(symbol_map[j])
            return results
    for j in list(symbol_map.values()):
        if param.upper() in j.upper():
            results.append(j)
    return results


def find_stock(params):
    results = []
    for j in params:
        tmp = find_stocks(j)
        results.append(tmp)
    for k in range(len(results) - 1):
        for j in results[k + 1]:
            if j not in results[k]:
                results[k + 1].remove(j)
    return results


pattern_match = {'symbol:(.*)-|symbol:(.*)': 'symbol',
                 'name:(.*)-|name:(.*)': 'name',
                 'description:(.*)-|description(.*)': 'description'}

rules = ['symbol:(.*)-|symbol:(.*)', 'name:(.*)-|name:(.*)', 'description:(.*)-|description(.*)']


def match_rule(message):
    global rules
    global pattern_match
    patterns = []
    sens = []
    for pattern in rules:
        match = re.search(pattern, message)
        if match is not None:
            if match.group(1) is not None:
                var = match.group(1)
                patterns.append(pattern_match[pattern])
                sens.append(var)
            else:
                var = match.group(0).replace(pattern_match[pattern] + ':', '')
                patterns.append(pattern_match[pattern])
                sens.append(var)
    if len(patterns) > 0:
        print(patterns, sens)
        return patterns, sens
    return "default", None


def check_stock(params):
    global state
    global lack_responses
    global params_stock
    global params_description
    stocks = find_stock(params)
    if len(stocks) == 0:
        state = FILLING
        tmp=params_stock[len(params_stock)-1]
        params_stock.clear()
        return "cannot_find",tmp
    else:
        k = len(stocks)
        stock = stocks[k - 1]
        if (len(stock)) > 1:
            state = FILLING
            return 'more_info'
        elif len(stock) == 1:
            state = FILLED
            return stock[0]
        elif len(stock) == 0:
            state = FILLING
            tmp=params_stock[len(params_stock)-1]
            params_stock.pop()
            return 'cannot_find',tmp


def modi_des(message):
    global nlp
    global descriptions
    global X
    text_message = message
    text_x = nlp(text_message).vector
    scores = [cosine_similarity(X[j, :].reshape(1, -1), text_x.reshape(1, -1))[0, 0] for j in range(len(descriptions))]
    index = scores.index(max(scores))
    result = des_modified[index]
    return result


def wait_affirm(message):
    global stockk
    global affirmative
    global wait_for_affirm
    global symbol_map_reverse
    global state
    global params_description
    global Token
    if affirmative == 1:
        wait_for_affirm = 0
        b, c = (stockk[len(stockk) - 2], stockk[len(stockk) - 1])
        affirmative = 0
        stockk.clear()
        return random.choice(choices).format(b, c)
    else:
        stockk.pop()
        stockk.pop()
        results={}
        if len(stockk) == 1:
            print("affirm1")
            wait_for_affirm = 0
            affirmative = 0
            k = stockk[0]
            stockk.clear()
            stock_symbol = symbol_map_reverse[k]
            stock_o = Stock(stock_symbol, token=Token)
            stock_info = stock_o.get_quote()
            print(stock_info)
            for m in params_description:
                k = modi_des(m)
                results[k] = stock_info[k]
            state = FILLING
            params_description.clear()
            n = len(results)
            p=ans[n - 1].format((*list(results.keys())), params_stock[len(params_stock) - 1],
                                     (*list(results.values())))
            return p
        elif len(stockk) == 2:
            print("affirm2")
            affirmative = 1
            return wait_affirm(message)
        else:
            print("affirm3")
            wait_for_affirm = 1
            affirmative = 0
            k = random.choice(more_than_one)
            if "{0}" in k:
                wait_for_affirm = 1
                stocks = find_stock(params_stock)
                stockk += stocks[len(stocks) - 1]
                return k.format(stockk[len(stockk) - 2], stockk[len(stockk) - 1])
            else:
                return k


def state_filled(message):
    global state
    global lack_responses
    global params_stock
    global params_description
    global start
    global wait_for_affirm
    global affirmative
    global cannot_find
    global more_than_one
    global stockk
    global symbol_map_reverse

    stock = check_stock(params_stock)
    results = {}
    if stock == "more_info":
        state=FILLING
        k = random.choice(more_than_one)
        if "{0}" in k:
            wait_for_affirm = 1
            stocks = find_stock(params_stock)
            stockk += stocks[len(stocks) - 1]
            print(1)
            return k.format(stockk[len(stockk) - 2], stockk[len(stockk) - 1])
        else:
            print(2)
            return k
    elif 'cannot_find' in stock:
        state=FILLING
        return random.choice(cannot_find).format(stock[1])
    else:
        stock_symbol = symbol_map_reverse[stock]
        stock_o = Stock(stock_symbol, token=Token)
        stock_info = stock_o.get_quote()
        for m in params_description:
            k = modi_des(m)
            results[k] = stock_info[k]
        state = FILLING
        params_description.clear()
        n = len(results)
        return ans[n - 1].format((*list(results.keys())), params_stock[len(params_stock) - 1],
                                 (*list(results.values())))




def state_filling(message):
    global state
    global lack_responses
    global params_stock
    global params_description
    global start
    global wait_for_affirm
    global params_append
    global affirmative
    global cannot_understand
    global lack_stock
    global lack_description

    patterns, varss = match_rule(message)
    parse_data = interpret(message)
    entities = parse_data['entities']
    print(params_stock)
    for ent in entities:
        if ent['entity'] == 'stock' and ":" not in ent['value']:
            params_stock.append(ent['value'])
        if ent['entity'] == 'description' and ":" not in ent['value']:
            if ent['value'] not in params_description:
                params_description.append(ent['value'])
    print(params_stock)
    if patterns == 'default' and len(entities) == 0:
        return random.choice(cannot_understand)
    elif patterns != 'default':
        print(varss)
        for m in range(len(patterns)):
            if patterns[m] != 'description':
                params_append[patterns[m]].append(varss[m])
            else:
                if varss[m] not in params_append[patterns[m]]:
                    params_append[patterns[m]].append(varss[m])
        print(params_stock)
    lack_symbol = 1
    lack_des = 2
    if len(params_description) > 0:
        lack_des = 0
    if len(params_stock) > 0:
        lack_symbol = 0
    lack = lack_des + lack_symbol
    if lack != 0:
        if lack == 1:
            k = random.choice(lack_stock)
            if "{0}" in k:
                return k.format(params_description[len(params_description) - 1])
            else:
                return k
        if lack == 2:
            k = random.choice(lack_description)
            if "{0}" in k:
                return k.format(params_stock[len(params_stock) - 1])
            else:
                return k
    else:
        state = FILLED
        return state_filled(message)


def begin(message):
    global state
    global lack_responses
    global params_stock
    global start
    global wait_for_affirm
    global affirmative
    global Help
    global stockk
    message=message.lower()
    data = interpreter.parse(message)
    intent = data['intent']['name']
    if intent == "thank":
        state = 0
        erase_all()
        return random.choice(thanks)
    if intent == 'bye':
        state = 0
        erase_all()
        return random.choice(Goodbye)
    elif intent == "what":
        return random.choice(Help)
    elif intent == 'greet':
        return random.choice(Greetings)
    elif state == 2 and start == 1 and wait_for_affirm == 0:
        return state_filled(message)
    elif state == 1 and start == 1 and wait_for_affirm == 0:
        return state_filling(message)
    elif intent == "affirmative" and wait_for_affirm==1:
        print("affrimative")
        affirmative=1
        return wait_affirm(message)
    elif intent == "negative" and wait_for_affirm==1:
        print("negative")
        affirmative=2
        return wait_affirm(message)
    elif intent == "query":
        stockk.clear()
        start = 1
        parse_data = interpret(message)
        entities = parse_data['entities']
        for ent in entities:
            if ent['entity'] == 'stock':
                params_stock.append(ent['value'])
            if ent['entity'] == 'description':
                params_description.append(ent['value'])
        lack_symbol = 1
        lack_des = 2
        if len(params_description)> 0:
            lack_des = 0
        if len(params_stock) > 0:
            lack_symbol = 0
        lack = lack_des + lack_symbol
        if lack == 0:
            state = FILLED
            return state_filled(message)
        else:
            state = FILLING
            return state_filling(message)


# Default statement
Greetings = [
    "Hello,I am Stark. I can help you find information of stocks. Simply send me your query like 'I want to know the open price of Apple.'"]
Help = [
    "Simply tell me what you are looking for like 'I want to know the market cap of facebook.' To give more information for more accurate result, you may use 'symbol:FB' or 'name:facebook' and 'description: open price'.To join them together, you can use '-' like 'symbol:FB-description:open price'. By using /next or saying 'thanks', you can end up this turm of search. You can use /help to check this message again."]
lack_stock = ["Sorry,I'm wondering which stock you are asking about.",
              "Could you tell me more about the stock you are interested in?", "You wanna learn the {0} of what stock?",
              "Sorry, I failed to understand the stock you are referring to. Please give me more information."]
lack_description = ["So what do you want to know about {0},", "Gotcha! Just tell me more about what you learn about it!"]
lack_both = ['Please tell me what you want to know']
cannot_find = [
    "Sorry, I cannot find the stock {0}. You may try again by telling me the iex symbol of the stock or other stocks.",
    "Sorry,I failed to find stock {0}. Please try other names of it or give me the symbol.",
    "I've never heard of {0}, did you type it wrong?"]
more_than_one = ["Does {0} or {1} fit your description? Or you wanna other results?",
                 "{0} and {1} are two results, do them meet your requirements. If not, I still have other answers.",
                 "there are too many options, I need more information to narrow it down",
                 "I get confused by such a lot of results. Please give me more information."]
cannot_understand = ["Sorry, I cannot understand you, please use /help to check how to make queries"]
thanks = ["You are welcome!", "My pleasure", " Anytime!", "It's the least I can do"]
Goodbye = ["See long!", 'Have a nice day!', 'Be seeing you!', "Catch you later"]
ans = ["The {0} of {1} is {2} USD.", "The {0} of {2} is {3} USD and the {1} of {2} is {4} USD",
       "The {0} of {2} is {4}, the {1} of {2} is {5} USD and the {3} of {2} is {6} USD."]
choices = ["{0} or {1}, what do you want to learn about? Please tell me it's full name."]


# Word_vector preparing
nlp = spacy.load("en_core_web_md")
descriptions = ['latestPrice', 'price', 'highest price', 'lowest price', 'Volume', 'latest volume', 'marketCap',
                'market cap', 'market vlue', 'Capitalization', 'open price', 'close price']
des_modified = ['latestPrice', 'latestPrice', 'high', 'low', 'latestVolume', 'latestVolume', 'marketCap', 'marketCap',
                'marketCap', 'marketCap', 'open', 'close']
# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length
n_description = len(descriptions)
X = np.zeros((n_description, embedding_dim))
# Iterate over the sentences
for idx, sentence in enumerate(descriptions):
    doc = nlp(sentence)
    X[idx, :] = doc.vector


# Initialize the empty dictionary lists and variables
state = 0
stockk = []
start = 0
INIT = 0
FILLING = 1
FILLED = 2
lack_responses = []
wait_for_affirm = 0
affirmative = 0
params_stock, params_description, params_time = [], [], []
params_append = {'symbol': params_stock, 'name': params_stock, 'description': params_description}


# Create a Rasa_NLU systiom
# Create a trainer that uses this config
trainer = Trainer(config.load("./training_data/config_spacy.yml"))
# Load the training data
training_data = load_data('./training_data/rasa_dataset_training.json')
# Create an interpreter by training the model
interpreter = trainer.train(training_data)


# mapping stocks' symbols to their names
Token = IEXTOKEN
a = get_symbols(token=Token)
symbol_map = {}
symbol_map_reverse = {}
for i in a:
    symbol_map[i['symbol']] = i['name']
    symbol_map_reverse[i['name']] = i['symbol']


# Initiate a telegram robot
bot = telebot.TeleBot(TOKEN)
apihelper.proxy = {'https': 'socks5h://127.0.0.1:1080'}
erase_all()


@bot.message_handler(commands=['start'])
def send_welcome(message):
    msg=Greetings[0]
    bot.send_message(message.chat.id, msg)


@bot.message_handler(commands=['next'])
def send_welcome(message):
    erase_all()


@bot.message_handler(commands=['help'])
def send_welcome(message):
    global Help
    bot.send_message(message.chat.id, Help[0])


@bot.message_handler()
def echo(message):
    msg = begin(message.text)
    bot.send_message(message.chat.id, msg)


# Start the bot
if __name__ == '__main__':
    bot.polling()