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
    access_token_secret = 'jlkuarq7ghrVEo6H9krFIc4j3Kq7zIp2leVhFz25l12LN'),

    # DS Cba
    dict(consumer_key = 'PGJgnuiRPw6ZiJrzVa7bpkcwg',
    consumer_secret = '91oIGr4hxcJviAEHFlbwwX79fHsA9aR3KX4nmVCdtVtLR6hMxV',
    access_token = '3072785271-raDCcTEGrtNJqJqLAn9LYJtL7Sz7pgEB67uMSfF',
    access_token_secret = 'wqlUE7tfstOyi8TuUkzT1q99e4xMtkchvEpfh0Kze35X7'),

    # tesiscomp0
    dict(consumer_key = 'W0i9r14YUAB4PGQHfp68X3SgY',
    consumer_secret = 'RykpfbJHsW44li88pDHy2pkjbrJorSmjDePSxjR8VtU7myAoM4',
    access_token = '393285785-DYfrJEDo2gUTRbYtlf3jPioZMaNDBlUPnzcoAFuG',
    access_token_secret = 'qeFL5xSxoU76dfKxLgJpbe7OQO5kyXHKM6New5fr1ljVV'),

    # tesiscomp1
    dict(consumer_key = 'f0MiYq2S5iLktvJf0sTfRsLLS',
    consumer_secret = 'xlvnp6fVAzlfpdShTICRPKbIT6YVlZUd9bofZG1lSEEMqP4pPn',
    access_token = '393285785-4yzNnw5EQU4VQVg60zczJjUHAWDN3M9SQOVos0aD',
    access_token_secret = 'yO0FuPRGYNgOUjjCGC7mTdGfnpKfyaH9ruTEZxzO5vFiC'),

    # tesiscomp2
    dict(consumer_key = 'VAetOVbpVSq75f7w50P3W0CSR',
    consumer_secret = 'z6J7yFJLeG0LzGWMDfdrG2GReavKNvV2gScyPu5S9AeTU6J3Iv',
    access_token = '393285785-f6bjB9LZWAcKIcBTmHaBuKGLeRBN9sq0vWWq4MBn',
    access_token_secret = 'fewVhQ0UulxCRuPyvkIOBHHLLtBUf3PR1gAFIRBlAn9fs'),

    # ds_tesiscomp0
    dict(consumer_key = 'QXRzocZPEZHt2J1nMdVd9mbeF',
    consumer_secret = 'zltlvMscQDGoQCX5UOHrK6BHmyfoYvGPUELuEmTlGhMAE1ZEI7',
    access_token = '3072785271-fWvbm54Z4efwRG54NskULSnHweiBgHdZUclYiP7',
    access_token_secret = 'Uu42AYIL2sPd9V9He9estF31GbcoeYhadVly1s70eGBtt'),

    # ds_tesiscomp1
    dict(consumer_key = 'QOlBFygFTecPH0W8UJdwNutqr',
    consumer_secret = 'XhCUoiSHdVOSAAiHrg3sf4zIF5xIUAexV3EIfbFMeVIHjAyLFi',
    access_token = '3072785271-GReaaepikVW8NXVtRzZrgNlqiQwNRB3GvTse167',
    access_token_secret = 'cCFyenr0B9LLwQyB9EePRW0oC7KdXxu3ApY7UgQ4RcXcc'),

    # ds_tesiscomp2
    dict(consumer_key = '2n7yeboGqZ3ldwXr08wOGE5wQ',
    consumer_secret = '7CFURPoJDhJKNb6VWVamAyS3WFKRrt556jJHsWifoDZBY41yfm',
    access_token = '3072785271-dWAo8qpOaEZ6J7IzPWDVd4nk3vC138nM2cNORJn',
    access_token_secret = 'AJNrwJUfoZinAgQz0HWA3fTNM1xVMgYL4aGluWYKRLLqo'),

]

USER_DATA = {
    'id': 393285785,
    'screen_name': 'PCelayes'
}
    
SEARCH_STRING = "the OR a"

N_USERS = 50

from local_settings import *