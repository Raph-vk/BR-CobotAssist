import os
import sys
import time
import torch
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Lock, Event, shared_memory

from .rl_config import RLConfig
from .act_config import ACTConfig
from .resip_threaded import ResiP
from .detectron import BallDetector
# from .detectron import BallDetector


def setup_logger(verbose_logging):
    logger = logging.getLogger("Main Process")
    if verbose_logging:
        logging.basicConfig(level=logging.INFO, format='[%(process)d %(name)s] %(message)s')
    else:
        logging.basicConfig(level=logging.CRITICAL)
    return logger

def run_resip(rl_cfg, shared_memories, shm_lock, stop_resip_event, interrupt_detectron_event, ckpt_path=None):
    resip = ResiP(rl_cfg, shared_memories, shm_lock, stop_resip_event, interrupt_detectron_event, ckpt_path)
    resip.run()

def create_shm_and_tensor(name, shape, dtype):
    nbytes = np.zeros(shape, dtype=dtype).nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np_array[:] = 0
    return shm, np_array

def resip_setup():
    act_cfg = ACTConfig()   # To-do load pre-created act config
    rl_cfg = RLConfig()
    rl_cfg.act = act_cfg

    logger = setup_logger(rl_cfg.verbose_logging)

    qpos_shm, qpos_np = create_shm_and_tensor('qpos_shm', rl_cfg.qpos_shape, np.float32)
    vision_shm, vision_np = create_shm_and_tensor('vision_shm', (rl_cfg.num_cameras, *rl_cfg.vision_shape), np.float32)
    next_base_action_shm, next_base_action_np = create_shm_and_tensor('next_base_action_shm', rl_cfg.action_shape, np.float32)
    final_action_shm, final_action_np = create_shm_and_tensor('final_action_shm', rl_cfg.action_shape, np.float32)

    predict_action_flag_shm, predict_action_flag_np = create_shm_and_tensor('predict_action_flag_shm', rl_cfg.flag_shape, np.uint8)
    action_ready_flag_shm, action_ready_flag_np = create_shm_and_tensor('action_ready_flag_shm', rl_cfg.flag_shape, np.uint8)
    activate_detectron_flag_shm, activate_detectron_flag_np = create_shm_and_tensor('activate_detectron_flag_shm', rl_cfg.flag_shape, np.uint8)
    reward_flag_shm, reward_flag_np = create_shm_and_tensor('reward_flag_shm', rl_cfg.flag_shape, np.uint8)
    env_reset_flag_shm, env_reset_flag_np = create_shm_and_tensor('env_reset_shm', rl_cfg.flag_shape, np.uint8)

    
    shm_lock = Lock()
    shared_memories = [qpos_shm, vision_shm, next_base_action_shm, final_action_shm,
                        predict_action_flag_shm, action_ready_flag_shm, activate_detectron_flag_shm, reward_flag_shm, env_reset_flag_shm]
    
    interrupt_detectron_event = Event()
    detectron_process = Process(target=BallDetector, args=(rl_cfg, activate_detectron_flag_shm, reward_flag_shm, shm_lock, interrupt_detectron_event), daemon=False)
    detectron_process.start()

    stop_resip_event = Event()
    resip_process = Process(target=run_resip, args=(rl_cfg, shared_memories, shm_lock, stop_resip_event, interrupt_detectron_event), daemon=True)
    resip_process.start()

    return rl_cfg, shared_memories, shm_lock, stop_resip_event, interrupt_detectron_event, detectron_process, resip_process   
    
def runner(logger, rl_cfg, shared_memories, shm_lock, stop_resip_event, interrupt_detectron_event, detectron_process, resip_process):
    (   
        qpos_shm,
        vision_data_shm,
        next_base_action_shm,
        final_action_shm,
        predict_action_flag_shm,
        action_ready_flag_shm,
        activate_detectron_flag_shm,
        reward_flag_shm,
        env_reset_flag_shm
    ) = shared_memories
    
    try:
        while True:
            time.sleep(0.000001)

            logger.info(f"Set reward, next done, and new obs and observation flag")

            while True:
                with shm_lock:
                    env_reset_flag = np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, buffer=env_reset_flag_shm.buf)[0]
                    action_ready_flag = np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, buffer=action_ready_flag_shm.buf)[0]
                if (env_reset_flag):
                    break
                if action_ready_flag:
                    with shm_lock:
                        np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, buffer=action_ready_flag_shm.buf)[0] = 0
                        next_action = torch.from_numpy(np.copy(np.ndarray(rl_cfg.action_shape, dtype=np.float32, buffer=final_action_shm.buf)))
                        # To-do implement action adding to array
                    break
                time.sleep(0.001)

            if (env_reset_flag):
                logger.info("Resetting Environment")

                with shm_lock:
                    interrupt_detectron_event.set()
                    np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, buffer=predict_action_flag_shm.buf)[0] = 0
                    np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, buffer=action_ready_flag_shm.buf)[0] = 0
                    np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, buffer=activate_detectron_flag_shm.buf)[0] = 0
                    np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, buffer=reward_flag_shm.buf)[0] = 0
                    np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, buffer=env_reset_flag_shm.buf)[0] = 0

                time.sleep(3)   # To-do implement reset
                continue

            logger.info(f"Next action is: {next_action}")

    except KeyboardInterrupt:
        logger.info("Exiting...")

        detectron_process.terminate()
        detectron_process.join()
        logger.info("Shutdown Detectron Process")

        stop_resip_event.set()
        resip_process.join()
        logger.info("Shutdown ResiP Process")

        for shm in shared_memories:
            shm.close()
            shm.unlink()
        logger.info("Closed shared memories")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    resip_setup()