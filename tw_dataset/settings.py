from datetime import date, timedelta

DATE_LIMIT = date.today() - timedelta(days=30)

AUTH_DATA = [
    # Mine
    dict(consumer_key = '09tNWUcdYl8pHIM80kDTzw8YI',
    consumer_secret = '4j5fMDyjLw78aYoSSLmPyzwuuR79pwQTbGZb1MpIPXH7DqqCPn',
    access_token = '393285785-1mhrJXFRJoOBRS0mHdTHw28b4vBBTTphhMpgfB3M',
    access_token_secret = 'BPaodd3EZaViweRr7FyA8F6xzLFJEwldYuMh7k3I1mg0y'),

    # Neil's
    dict(consumer_key = 'cTE7sd9pKZWaWGnSGDPdCTjsc',
    consumer_secret = 'WqcCgCrbltzCvUoLIlKR9boX3PHL6fff55v4Xk66MyBBHpPKPv',
    access_token = '998917016-yh7r3tZrIEB8Z9FcXO3N3OqnTmm353QfX855qwPq',
    access_token_secret = 'jlkuarq7ghrVEo6H9krFIc4j3Kq7zIp2leVhFz25l12LN')
]

USER_DATA = {
    'id': 393285785,
    'screen_name': 'PCelayes'
}

    
SEARCH_STRING = "the OR a"

WORDS_PER_USER = 500

N_USERS = 50

WAIT_BETWEEN_COUNTIES = 0

WAIT_BETWEEN_USERS = 0

