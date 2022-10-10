# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조
import logging
import re
from typing import Dict,Optional, Tuple

import flwr as fl
import tensorflow as tf
import tensorflow_addons as tfa

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

import wandb
import datetime
import os

import requests, json
import time

# Log 포맷 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# FL 하이퍼파라미터 설정
num_rounds = 5
local_epochs = 10
batch_size = 2048
val_steps = 5


# s3에 global model upload
def upload_model_to_bucket(global_model):
    global next_gl_model
    
    logging.info(f'gl_model_{next_gl_model}_V.h5 모델 업로드 시작')

    global_model.save('/gl_model/gl_model_%s_V.h5' % next_gl_model)

    logging.info(f'gl_model_{next_gl_model}_V.h5 모델 업로드 완료')


# s3에 저장되어 있는 latest global model download
def model_download():

    try:
        gl_list = os.listdir('./gl_model')
        del gl_list[1] # mac에서만 시행 (.DS_Store 파일 삭제)
        s = gl_list[0] # 비교 대상(gl_model 지정) => sort를 위함
        p = re.compile(r'\d+') # 숫자 패턴 추출
        gl_list_sorted = sorted(gl_list, key=lambda s: int(p.search(s).group())) # gl model 버전에 따라 정렬

        gl_model = gl_list_sorted[len(gl_list_sorted)-1] # 최근 gl model 추출
        gl_model_v = int(gl_model.split('_')[2])

        logging.info(f'gl_model: {gl_model}, gl_model_v: {gl_model_v}')

        return gl_model, gl_model_v

    # s3에 global model 없을 경우
    except Exception as e:
        logging.info(f'global model read error: {e}')

        model_X = 'null'
        gl_model_v = 0
        logging.info(f'gl_model: {model_X}, gl_model_v: {gl_model_v}')
        return model_X, gl_model_v


def fl_server_start(model):

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation    
    
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        ]

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=METRICS)


    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit > fraction_eval이여야 함
        # min_available_clients의 수를 실제 연결 client 수 보다 작게 하는게 안정적임
        # => client가 학습 중에 멈추는 현상이 가끔 발생
        fraction_fit=0.6, # 클라이언트 학습 참여 비율
        fraction_evaluate=0.5, # 클라이언트 평가 참여 비율
        min_fit_clients=2, # 최소 학습 참여 수
        min_evaluate_clients=2, # 최소 평가 참여 수
        min_available_clients=2, # 최소 클라이언트 연결 필요 수
        evaluate_fn=get_eval_fn(model), # 모델 평가 결과
        on_fit_config_fn=fit_config, # batchsize, epoch 수
        on_evaluate_config_fn=evaluate_config, # val_step
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def gl_model_load():

    global num_rounds, latest_gl_model_v
    # global x_val, y_val # f1_score 계산을 위해 label 개수 확인

    print('')
    logging.info(f'latest_gl_model_v: {latest_gl_model_v}')
    print('')
    
    if os.path.isfile('./gl_model/gl_model_%s_V.h5'%latest_gl_model_v):
        logging.info('new global model loading')
        model = tf.keras.models.load_model('./gl_model/gl_model_%s_V.h5'%latest_gl_model_v)
        fl_server_start(model)

    else:
        # global model 없을 시 초기 글로벌 모델 생성
        logging.info('init model making')

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

        fl_server_start(model)
        

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    global x_val, y_val

    # print('get_eval_fn x_val length: ', x_val.shape[-1])

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        
        # loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
        loss, accuracy = model.evaluate(x_val, y_val)

        global next_gl_model

        # model save
        model.save("./gl_model/gl_model_%s_V.h5"%next_gl_model)

        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """

    global batch_size, local_epochs

    config = {
        "batch_size": batch_size,
        # "local_epochs": 1 if rnd < 2 else 2,
        "local_epochs": local_epochs,
        "num_rounds": num_rounds,
    }

    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    # val_steps = 5 if rnd < 4 else 10
    global val_steps
    
    return {"val_steps": val_steps}


if __name__ == "__main__":
    
    today= datetime.datetime.today()
    today_time = today.strftime('%Y-%m-%d %H-%M-%S')

    # global model download
    model, latest_gl_model_v = model_download()

    # 새로 생성되는 글로벌 모델 버전
    next_gl_model = latest_gl_model_v + 1

    # server_status 주소
    inform_SE: str = 'http://0.0.0.0:8050/FLSe/'

    inform_Payload = {
            # 형식
            'S3_bucket': 'fl-flower-model', # 버킷명
            'S3_key': 'gl_model_%s_V.h5'%latest_gl_model_v,  # 모델 가중치 파일 이름
            'play_datetime': today_time, # server 수행 시간
            'FLSeReady': True, # server 준비 상태 on
            'GL_Model_V' : latest_gl_model_v # GL 모델 버전
        }

    while True:
        try:
            # server_status => FL server ready
            r = requests.put(inform_SE+'FLSeUpdate', verify=False, data=json.dumps(inform_Payload))
            if r.status_code == 200:
                break
            else:
                logging.info(f'{r.content}')
        except:
            logging.info("Connection refused by the server..")
            time.sleep(5)
            continue

    # Cifar 10 데이터셋 불러오기
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
    num_classes = 10	

    # global model 평가를 위한 데이터셋
    x_val, y_val = X_test[1000:9000], y_test[1000:9000]

    # y(label) one-hot encoding
    y_val = to_categorical(y_val, num_classes)

    try:
        fl_start_time = time.time()

        # Flower Server 실행
        gl_model_load()

        fl_end_time = time.time() - fl_start_time  # 연합학습 종료 시간
        fl_server_operation_time = str(datetime.timedelta(seconds=fl_end_time)) # 연합학습 종료 시간
        logging.info(f'fl_server_operation_time: {fl_server_operation_time}')

        # 버킷에 global model upload
        # upload_model_to_bucket("gl_model_%s_V.h5" %next_gl_model)

        
    # FL server error
    except Exception as e:
        logging.info(f'error: {e}')
        data_inform = {'FLSeReady': False}
        requests.put(inform_SE+'FLSeUpdate', data=json.dumps(data_inform))
        
    finally:
        logging.info('server close')
      
        # server_status에 model 버전 수정 update request
        res = requests.put(inform_SE + 'FLRoundFin', params={'FLSeReady': 'false'})
        if res.status_code == 200:
            new_gl_model_v = res.json()['Server_Status']['GL_Model_V']
            logging.info('global model version upgrade')
            logging.info(f'new global model version: {new_gl_model_v}')
