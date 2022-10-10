import argparse

from pydantic.main import BaseModel
import logging
import requests
import uvicorn
from fastapi import FastAPI
import asyncio
import wget
from functools import partial
from datetime import datetime, timedelta 

from functools import wraps

import requests

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
app = FastAPI()

# 날짜를 폴더로 설정
global today_str
today= datetime.today()
today_str = today.strftime('%Y-%m-%d')

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--number", type=int, choices=range(0, 10), required=True)
args = parser.parse_args()
number = args.number # client 번호

class manager_status(BaseModel):
    global today_str, number

    # INFER_SE: str = '0.0.0.0:8001'
    FL_client: str = '127.0.0.1:800%s'%number
    FL_server_ST: str = '0.0.0.0:8050'
    FL_server: str = '0.0.0.0:8080'  # '0.0.0.1:8080'
    FL_client_num: int = 0
    GL_Model_V: int = 0  # 모델버전
    FL_ready: bool = False  # FL server준비됨
    have_server_ip: bool = True  # server 주소가 확보되어있음

    FL_client_online: bool = False  # flower client online?
    FL_learning: bool = False  # flower client 학습중


manager = manager_status()


@app.on_event("startup")
def startup():
    ##### S0 #####
    
    get_server_info()

    ##### S1 #####
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    # 전역변수값을 보고 상태를 유지하려고 합니다.
    # 이런식으로 짠 이유는 개발과정에서 각 구성요소의 상태가 불안정할수 있기 때문으로
    # manager가 일정주기로 상태를 확인하고 또는 명령에 대한 반환값을 가지고 정보를 갱신합니다
    loop.create_task(health_check())
    # loop.create_task(check_infer_online())
    loop.create_task(check_flclient_online())
    # loop.create_task(infer_update())
    loop.create_task(start_training())


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/trainFin")
def fin_train():
    global manager
    print('fin')
    # manager.infer_ready = True
    manager.FL_learning = False
    manager.FL_ready = False
    manager.GL_Model_V += 1
    return manager


@app.get("/trainFail")
def fail_train():
    global manager
    print('Fail')
    #manager.infer_ready = False
    manager.FL_learning = False
    manager.FL_ready = False
    asyncio.run(health_check()) 
    return manager


@app.get('/info')
def get_manager_info():
    return manager

@app.get('/flclient_out')
def flclient_out():
    manager.FL_client_online = False
    manager.FL_learning = False
    return manager


def async_dec(awaitable_func):
    async def keeping_state():
        while True:
            try:
                logging.debug(str(awaitable_func.__name__) + '함수 시작')
                # print(awaitable_func.__name__, '함수 시작')
                await awaitable_func()
                logging.debug(str(awaitable_func.__name__) + '_함수 종료')
            except Exception as e:
                # logging.info('[E]' , awaitable_func.__name__, e)
                logging.error('[E]' + str(awaitable_func.__name__) + str(e))
            await asyncio.sleep(1)

    return keeping_state


@async_dec
async def health_check():
    global manager
    print('FL_learning: ', manager.FL_learning)
    print('FL_client_online: ', manager.FL_client_online)
    print('FL_ready: ', manager.FL_ready)
    if (manager.FL_learning == False) and (manager.FL_client_online == True):
        loop = asyncio.get_event_loop()
        # raise
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_server_ST + '/FLSe/info'))
        if (res.status_code == 200) and (res.json()['Server_Status']['FLSeReady']):
            # if res.json()['Server_Status']['GL_Model_V'] != manager.GL_Model_V:
            #     await pull_model()
            #     manager.GL_Model_V = res.json()['Server_Status']['GL_Model_V']
            manager.FL_ready = res.json()['Server_Status']['FLSeReady']
            # logging.info('flclient learning')
            # manager.FL_learning = True
        elif (res.status_code != 200):
            # manager.FL_client_online = False
            logging.error('FL_server_ST offline')
            # exit(0)
        else:
            pass
    else:
        pass
    await asyncio.sleep(8)

@async_dec
async def check_flclient_online():
    global manager
    if (manager.FL_client_online == False):
        logging.info('FL_client offline')
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_client + '/online'))
        if (res.status_code == 200) and (res.json()['FL_client_online']):
            manager.FL_client_online = res.json()['FL_client_online']
            manager.FL_learning = res.json()['FL_client_start']
            manager.FL_client_num = res.json()['FL_client']
            logging.info('FL_client online')
        else:
            logging.info('FL_client offline')
            pass
    else:
        await asyncio.sleep(12)


@async_dec
async def start_training():
    global manager
    if (manager.FL_client_online == True) and (manager.FL_learning == False) and (manager.FL_ready == True):
        logging.info('start training')
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, requests.get, ('http://' + manager.FL_client + '/start/'+manager.FL_server))
        if (res.status_code == 200) and (res.json()['FL_client_start']):
            manager.FL_learning = True
            logging.info('start_train')
        elif (res.status_code != 200):
            manager.FL_client_online = False
            logging.info('flclient offline')
        else:
            pass
    else:
        await asyncio.sleep(13)


def get_server_info():
    global manager
    try:
        res = requests.get('http://' + manager.FL_server_ST + '/FLSe/info')
        # print(res.json())
        manager.GL_Model_V = res.json()['Server_Status']['GL_Model_V']
    except Exception as e:
        raise e
    return manager


if __name__ == "__main__":
    # asyncio.run(training())
    port = int('900%s' % number)

    uvicorn.run("app:app", host='0.0.0.0', port=port, reload=True)