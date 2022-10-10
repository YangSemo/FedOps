# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조
import datetime
import os, logging, json
import argparse
import time

import tensorflow as tf

import flwr as fl
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
# keras에서 내장 함수 지원(to_categofical())
from keras.utils.np_utils import to_categorical

from functools import partial
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel

# Log 포맷 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Make TensorFlow logs less verbose
# TF warning log 필터링
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # CPU만 사용
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 성능
loss = 0
accuracy = 0

next_gl_model = 0

round = 1 # 현재 수행 round 수


# FL client 상태 확인
app = FastAPI()

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--number", type=int, choices=range(0, 10), required=True)
args = parser.parse_args()
client_num = args.number


# FL Client 상태 class
class FLclient_status(BaseModel):
    FL_client_num: int = client_num # FL client 번호(ID)
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None # FL server IP
    FL_round: int = 1 # 현재 수행 round
    FL_loss: int = 0 # 성능 loss
    FL_accuracy: int = 0 # 성능 acc
    FL_next_gl_model: int = 0 # 글로벌 모델 버전


status = FLclient_status()


# Define Flower client
class CifarClient(fl.client.NumPyClient):

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def get_properties(self, config):
        # fl_status = fl.common.Status(code=fl.common.Code.OK, message="Success")
        # properties = {"mse": 0.5} # super().get_properties({"mse": 0.5})
        # return fl.common.PropertiesRes(fl_status, properties)
        """Get properties of client."""
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        global status

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        # num_rounds: int = config["num_rounds"]

        # round 시작 시간
        round_start_time = time.time()

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        # round 종료 시간
        round_end_time = time.time() - round_start_time  # 연합학습 종료 시간
        round_client_operation_time = str(datetime.timedelta(seconds=round_end_time))
        logging.info(f'round: {status.FL_round}, round_client_operation_time: {round_client_operation_time}')

        # 다음 라운드 수 증가
        status.FL_round += 1

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        status.FL_loss = history.history["loss"][0]
        status.FL_accuracy = history.history["accuracy"][0]
        results = {
            "loss": status.FL_loss,
            "accuracy": status.FL_accuracy,
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)

        print('test_loss: ', test_loss, 'test_accuracy: ', test_accuracy)

        return test_loss, num_examples_test, {"accuracy": test_accuracy}


@app.on_event("startup")
def startup():
    pass


@app.get('/online')
def get_info():
    return status


@app.get("/start/{Server_IP}")
async def flclientstart(background_tasks: BackgroundTasks, Server_IP: str):
    global status

    # client_manager 주소
    client_res = requests.get('http://localhost:900%s/info/' % status.FL_client_num)

    # 최신 global model 버전
    latest_gl_model_v = client_res.json()['GL_Model_V']

    # 다음 global model 버전
    status.FL_next_gl_model = latest_gl_model_v + 1

    logging.info('FL client start')
    status.FL_client_start = True
    status.FL_server_IP = Server_IP
    background_tasks.add_task(flower_client_start)

    return status


# Client Local Model 생성
def build_model():
    # 모델 및 메트릭 정의
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]

    # model 생성
    model = Sequential()

    # Convolutional Block (Conv-Conv-Pool-Dropout)
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classifying
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=METRICS)

    return model


async def flower_client_start():
    logging.info('FL learning ready')
    global model, status

    # 환자별로 partition 분리 => 개별 클라이언트 적용
    (x_train, y_train), (x_test, y_test) = load_partition()
    # await asyncio.sleep(30) # data download wait
    logging.info('data loaded')

    model = build_model()
    logging.info('bulid model')

    try:
        loop = asyncio.get_event_loop()
        client = CifarClient(model, x_train, y_train, x_test, y_test)
        # logging.info(f'fl-server-ip: {status.FL_server_IP}')
        # await asyncio.sleep(23)
        print('server IP: ', status.FL_server_IP)
        request = partial(fl.client.start_numpy_client, server_address=status.FL_server_IP, client=client)

        # 라운드 수 초기화
        status.FL_round = 1

        fl_start_time = time.time()  # 연합학습 초기 시작 시간

        await loop.run_in_executor(None, request)  # 연합학습 Client 비동기로 수행

        logging.info('fl learning finished')

        fl_end_time = time.time() - fl_start_time  # 연합학습 종료 시간
        fl_client_operation_time = str(datetime.timedelta(seconds=fl_end_time))
        logging.info(f'fl_client_operation_time: {fl_client_operation_time}')

        await model_save()
        logging.info('model_save')

        # client 객체 및 fl_client_start request 삭제
        del client, request

    except Exception as e:
        logging.info('[E][PC0002] learning', e)
        status.FL_client_fail = True
        await notify_fail()
        status.FL_client_fail = False
        raise e

    return status


async def model_save():
    global model, status
    try:
        model.save('./local_model/num_%s_local_model_V%s.h5' % (status.FL_client_num, status.FL_next_gl_model))
        await notify_fin()
        model = None
    except Exception as e:
        logging.info('[E][PC0003] learning', e)
        status.FL_client_fail = True
        await notify_fail()
        status.FL_client_fail = False

    return status


# client manager에서 train finish 정보 확인
async def notify_fin():
    global status

    status.FL_client_start = False

    # 최종 성능 결과
    result = {"loss": status.FL_loss, "accuracy": status.FL_accuracy, "next_gl_model": status.FL_next_gl_model}
    json_result = json.dumps(result)
    print(json_result)
    print('{"client_num": ' + str(status.FL_client_num) + ', "log": "' + str(json_result) + '"}')

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:900%s/trainFin' % status.FL_client_num)
    r = await future2
    print('try notify_fin')
    if r.status_code == 200:
        print('trainFin')
    else:
        print('notify_fin error: ', r.content)
    return status


# client manager에서 train fail 정보 확인
async def notify_fail():
    global status

    logging.info('notify_fail start')

    status.FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:900%s/trainFail' % status.FL_client_num)
    r = await future1
    if r.status_code == 200:
        logging.info('trainFin')
    else:
        logging.info('notify_fail error: ', r.content)

    return status


def load_partition():
    # Load the dataset partitions
    global status

    # Cifar 10 데이터셋 불러오기
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # client_num 값으로 데이터셋 나누기
    (X_train, y_train) = X_train[status.FL_client_num * 100:(status.FL_client_num + 1) * 500], y_train[
                                                                           status.FL_client_num * 100:(status.FL_client_num + 1) * 500]
    (X_test, y_test) = X_test[status.FL_client_num * 100:(status.FL_client_num + 1) * 500], y_test[status.FL_client_num * 100:(status.FL_client_num + 1) * 500]

    # class 설정
    num_classes = 10

    # one-hot encoding class 범위 지정
    # Client마다 보유 Label이 다르므로 => 전체 label 수를 맞춰야 함
    train_labels = to_categorical(y_train, num_classes)
    test_labels = to_categorical(y_test, num_classes)

    # 전처리
    train_features = X_train.astype('float32') / 255.0
    test_features = X_test.astype('float32') / 255.0

    return (train_features, train_labels), (test_features, test_labels)


if __name__ == "__main__":

    try:
        # client api 생성 => client manager와 통신하기 위함
        uvicorn.run("app:app", host='0.0.0.0', port=int('800%s' % client_num), reload=True)

    finally:
        # FL client out
        requests.get('http://localhost:800%s/flclient_out' % client_num)
        logging.info('%s client close' % client_num)
