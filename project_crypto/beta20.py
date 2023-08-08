import requests
from datetime import date, datetime
import openpyxl

def consuming_api():
    coin = 'coti'
    list_coin = ['coti', 'gala', 'bitcoin', 'cardano', 'polkadot', 'hedera-hashgraph', 'tron', 'decentraland', 'enjin-coin', 'fantom']
    for coin in list_coin:
        while True:
            try:
                request = requests.get(f'http://api.coincap.io/v2/assets/{coin}')
                data = request.json()
                break
            except Exception as error:
                continue
        print(data)

        aux = str(data['timestamp'])[:-3]

        dt = datetime.fromtimestamp(float(aux))
        week = date(int(dt.year), int(dt.month), int(dt.day)).isocalendar()[1]

        info = [data['data']['symbol'], week, str(dt)[:11], data['data']['priceUsd'], data['data']['changePercent24Hr'], data['data']['marketCapUsd'], data['data']['volumeUsd24Hr'], data['data']['supply'], data['data']['maxSupply'], data['data']['vwap24Hr'], str(request)]
        print(info)
        feed_excel(info)
    return 'Done'

def feed_excel(info):
    try:
        wb = openpyxl.load_workbook(r'C:\Users\lucas\Estudos\db_excel\db_crypto.xlsx')
        ws = wb.active
        ws.append(info)
        wb.save(r'C:\Users\lucas\Estudos\db_excel\db_crypto.xlsx')
        wb.close()
        return True
    except Exception as error:
        print(error)
        return False






print(consuming_api())
