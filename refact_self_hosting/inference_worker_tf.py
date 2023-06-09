import os
import time
import importlib
import traceback
import logging
import socket

from typing import Optional, Dict, Any, List

from code_contrast.modeling import CodeTFModel
from code_contrast.scratchpad import ScratchpadCodeTF
# from refact_self_hosting import env
from smallcloud import inference_server


inference_server.override_urls("http://127.0.0.1:8008/infengine-v1/")


log = logging.getLogger("MODEL").info


quit_flag = False
DEBUG = int(os.environ.get("DEBUG", "0"))
if DEBUG:
    inference_server.DEBUG_UPLOAD_NOT_SEPARATE_PROCESS = True


def modload(import_str):
    import_mod, import_class = import_str.rsplit(":", 1)
    model = importlib.import_module(import_mod)
    Class = getattr(model, import_class, None)
    return Class


class Inference:
    def __init__(
            self,
            model_name: str,
            task: str,
            model_type: str,
            is_eval: bool = True,
            load_in_8bit: bool = True,
            load_in_4bit: bool = False,
            weight_sharding: bool = False,
    ):
        self._model_name = model_name
        self._task = task
        self._model_type = model_type
        self._is_eval = is_eval
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._weight_sharding = weight_sharding

    def _model_setup(self) -> CodeTFModel:
        return CodeTFModel(
            model_name=self._model_name,
            task=self._task,
            model_type=self._model_type,
            is_eval=self._is_eval,
            load_in_8bit=self._load_in_8bit,
            load_in_4bit=self._load_in_4bit,
            weight_sharding=self._weight_sharding,
        )

    def _prepare_scratchpad(self, request: Dict[str, Any]):
        return ScratchpadCodeTF(
            sources=request['sources'],
            code_tf_model=self._model_setup(),
            cursor0=request['cursor0'],
            cursor1=request['cursor1'],
        )

    def _generate_using_scratchpad(self, scratchpad: ScratchpadCodeTF):
        return scratchpad.completion()

    def infer(
            self,
            request: Dict[str, Any],
            upload_proxy: inference_server.UploadProxy,
            upload_proxy_args: Dict
    ):
        request_id = request["id"]
        try:
            scratchpad = self._prepare_scratchpad(request)
            upload_proxy_args["ts_prompt"] = time.time()
            if request_id in upload_proxy.check_cancelled():
                scratchpad.finish_reason = "cancelled"
                return

            for idx, _ in enumerate(self._generate_using_scratchpad(scratchpad)):
                if idx == 0:
                    upload_proxy_args["ts_first_token"] = time.time()
                if scratchpad.needs_upload:
                    scratchpad.needs_upload = False
                upload_proxy.upload_result(
                    **upload_proxy_args,
                    files=[scratchpad.completion()],
                    finish_reason=[scratchpad.finish_reason],
                    more_toplevel_fields=[scratchpad.toplevel_fields()],
                    generated_tokens_n=[scratchpad.generated_tokens_n],
                    status="in_progress"
                )
            assert scratchpad.finish_reason
            if DEBUG:
                scratchpad.debuglog("finish_reason", scratchpad.finish_reason)
            upload_proxy_args["ts_batch_finished"] = time.time()
            upload_proxy.upload_result(
                **upload_proxy_args,
                files=[scratchpad.completion(True)],
                finish_reason=[scratchpad.finish_reason],
                more_toplevel_fields=[scratchpad.toplevel_fields()],
                generated_tokens_n=[scratchpad.generated_tokens_n],
                status="completed"
            )
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())


def worker_loop(
        model_name: str,
        task: str,
        model_type: str
):
    log("loading model '%s'" % model_name)
    inference_model = Inference(
        model_name=model_name,
        task=task,
        model_type=model_type
    )
    class DummyUploadProxy:
        def upload_result(*args, **kwargs):
            pass

        @staticmethod
        def check_cancelled():
            return set()
    dummy_calls = [
        {
            'temperature': 0.8, 'top_p': 0.95, 'max_tokens': 40, 'id': 'comp-wkCX57Le8giP-1337', 'object': 'text_completion_req',
            'function': 'completion',
            'echo': False,
            'stop_tokens': [],
            'sources': {"hello.py": "def hello_world():"},
            'cursor0': 3,
            'cursor1': 3,
            'created': time.time(),
        }
    ]
    log("running a test batch")
    inference_model.infer(dummy_calls[0], DummyUploadProxy, {})

    req_session = inference_server.infserver_session()
    model = f'{model_name}/{model_type}'.replace('-', '_')
    description_dict = inference_server.validate_description_dict(
        model + "_" + socket.getfqdn(),
        "account_name",
        model=model, B=1, max_thinking_time=10,
    )
    upload_proxy = inference_server.UploadProxy(
        upload_q=None, cancelled_q=None)
    upload_proxy.start_upload_result_daemon()

    while not quit_flag:
        upload_proxy.keepalive()
        upload_proxy.cancelled_reset()
        retcode, request_batch = inference_server.completions_wait_batch(
            req_session, description_dict, verbose=False)
        ts_arrived = time.time()
        if retcode == "OK":
            for request in request_batch:
                upload_proxy_args = {
                    "description_dict": description_dict,
                    "original_batch": [request],
                    "idx_updated": [0],
                    "tokens": None,
                    "ts_arrived": ts_arrived,
                    "ts_batch_started": time.time(),
                    "ts_prompt": 0,
                    "ts_first_token": 0,
                    "ts_batch_finished": 0,
                }
                inference_model.infer(request, upload_proxy, upload_proxy_args)
        elif retcode == "WAIT":
            # Normal, no requests
            pass
        else:
            # No connectivity, connection refused, other errors go there
            time.sleep(10)

    upload_proxy.stop()


if __name__ == "__main__":
    worker_loop(
        model_name='codet5',
        task='pretrained',
        model_type='plus-770M-python'
    )
