from binance.client import Client

import time
import json
import requests

client = Client('api_key', 'api_secret')
COINS = ['BTCUSDT', 'ETHUSDT', 'EOSUSDT', 'NEOUSDT', 'ADAUSDT', 'XRPUSDT','BNBUSDT']
coins = ['BTC', 'ETH', 'EOS', 'NEO', 'ADA', 'XRP', 'BNB', 'USDT']
lot_size_dict = {'BTCUSDT':6,'ETHUSDT':5,'EOSUSDT':2,'NEOUSDT':3,'ADAUSDT':1,'XRPUSDT':1,'BNBBTC':2,'BNBETH':2,'EOSBNB':2,'NEOBNB':3,'ADABNB':1,'XRPBNB':2}

lot_size_dict_list = ['BTCUSDT','ETHUSDT','EOSUSDT','NEOUSDT','ADAUSDT','XRPUSDT','BNBBTC','BNBETH','EOSBNB','NEOBNB','ADABNB','XRPBNB']

#for lis in lot_size_dict_list:
#    wait = float(client.get_symbol_info(lis)['filters'][1]['stepSize'])
#    for idx in range(1000):
#        wait = wait*10
#        if wait == 1:
#            print(idx+1)
#            break
#    time.sleep(1)
    
def get_price():
    for idx in range(3):
        try:
            prices = []
            get_ = client.get_all_tickers()
            
            for COIN_ in COINS:
                for idx,lis in enumerate(get_):
                    if lis['symbol'] == COIN_:
                        prices.append(float(lis['price']))
                        break
            break
        except:
            time.sleep(0.5)
    
    return prices

def get_balance():
    balance = []    
    info = client.get_account()

    for coin in coins:
        for idx,lis in enumerate(info['balances']):
            if lis['asset'] == coin:
                balance.append(float(lis['free']))
                break

    return balance

def market_buy_order(amount, COIN_):
    lot_size = lot_size_dict[COIN_]

    #calc    
    if round(amount,lot_size) - amount > 0:
        quantity = round(round(amount,lot_size) - round(pow(0.1,lot_size),lot_size), lot_size)
    else:
        quantity = round(amount,lot_size)
        
    #order
    order = client.order_market_buy(
        symbol=COIN_,
        quantity=quantity)
    time.sleep(0.1)
    
    return order

def market_sell_order(amount, COIN_):
    lot_size = lot_size_dict[COIN_]
    
    #calc    
    if round(amount,lot_size) - amount > 0:
        quantity = round(round(amount,lot_size) - round(pow(0.1,lot_size),lot_size), lot_size)
    else:
        quantity = round(amount,lot_size)
        
    #order
    order = client.order_market_sell(
        symbol=COIN_,
        quantity=quantity)
    time.sleep(0.1)

    return order

def last_deposit_return():
    btc_deposit = client.get_deposit_history(asset='BTC')['depositList'][0]['amount']
    
    return btc_deposit

def last_withdraw_return():
    btc_withdraw = client.get_withdraw_history(asset='BTC')['withdrawList'][0]['amount']

    return btc_withdraw

"""def Loop_market_buy_order(amount):
    loop_numb = int(amount // 0.03)
    min_amount = round(amount/(loop_numb), 8) - 0.00000001
    
    for idx in range(loop_numb):
        order = client.create_market_buy(
            product_id=5,
            quantity='{}'.format(min_amount))
        
        if idx != loop_numb -1:
            time.sleep(0.1)

    return order
    
def Loop_market_sell_order(amount):
    loop_numb = int(amount // 0.03)
    min_amount = round(amount/(loop_numb), 8) - 0.00000001
    
    for idx in range(loop_numb):
        order = client.create_market_sell(
            product_id=5,
            quantity='{}'.format(min_amount))
        
        if idx != loop_numb -1:
            time.sleep(0.1)
        
    return order"""


