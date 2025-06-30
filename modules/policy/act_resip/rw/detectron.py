import cv2
import time
import torch
import logging
import numpy as np
import pyrealsense2 as rs
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

def setup_logger(verbose_logging):
    logger = logging.getLogger("Detectron Process")
    if verbose_logging:
        logging.basicConfig(level=logging.INFO, format='[%(process)d %(name)s] %(message)s')
    else:
        logging.basicConfig(level=logging.CRITICAL)
    return logger

class BallDetector:
    def __init__(self, cfg, activate_detectron_flag_shm, reward_flag_shm, shm_lock, interrupt_detectron_event):
        self.cfg = cfg
        self.logger = setup_logger(getattr(cfg, "verbose_logging", True))
        self.logger.info("Initializing BallDetector")

        self.cfg.detectron_cfg = get_cfg()
        self.cfg.detectron_cfg.merge_from_file(model_zoo.get_config_file(self.cfg.detectron_model_name))
        self.cfg.detectron_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust if needed
        self.cfg.detectron_cfg.MODEL.RETINANET.NUM_CLASSES = 1
        self.cfg.detectron_cfg.MODEL.WEIGHTS = self.cfg.detectron_model_path
        self.cfg.detectron_cfg.MODEL.DEVICE = self.cfg.detectron_device
        self.cfg.detectron_cfg.MODEL.MASK_ON = False
        self.cfg.detectron_cfg.MODEL.KEYPOINT_ON = False
        self.predictor = DefaultPredictor(self.cfg.detectron_cfg)

        self.class_names = ["Ball"]
        MetadataCatalog.get("ball_dataset").thing_classes = self.class_names
        self.metadata = MetadataCatalog.get("ball_dataset")

        # self.pipeline = rs.pipeline()
        # self.pipeline.start()

        self.activate_detectron_flag_shm = activate_detectron_flag_shm
        self.reward_flag_shm = reward_flag_shm
        self.shm_lock = shm_lock
        self.interrupt_detectron_event = interrupt_detectron_event
        self.stopping = False

        self.run()

    def run(self):
        try:
            while True:
                time.sleep(100)
                while True:
                    # self.logger.info("Waiting for activation signal from ResiP...")
                    with self.shm_lock:
                        if np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.activate_detectron_flag_shm.buf)[0] == 1:
                            break
                        np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.activate_detectron_flag_shm.buf)[0] = 0
                    time.sleep(0.01)
                self.logger.info("Activation signal received, starting detection...")
                for _ in range(self.cfg.detectron_check_frames):                
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())

                    start_time = time.time()
                    with torch.no_grad():
                        outputs = self.predictor(color_image)
                    instances = outputs["instances"].to("cpu")
                    confidence_mask = instances.scores >= self.cfg.confidence_threshold
                    instances = instances[confidence_mask]

                    boxes = instances.pred_boxes.tensor.numpy()
                    for box in boxes:
                        x_min, y_min, x_max, y_max = box
                        x_center = (x_max + x_min) / 2
                        y_center = (y_max + y_min) / 2
                        depth = depth_frame.get_distance(int(x_center), int(y_center))
                        if (self.cfg.depth_min <= depth <= self.cfg.depth_max and
                            self.cfg.center_x_min <= x_center <= self.cfg.center_x_max and
                            self.cfg.center_y_min <= y_center <= self.cfg.center_y_max):
                            
                            if (not self.interrupt_detectron_event.is_set()):
                                with self.shm_lock:
                                    np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.reward_flag_shm.buf)[0] = 1
                                self.logger.info(f"Ball detected in the center in {round((time.time() - start_time) * 1000, 1)} ms")
                                self.stopping = True
                            break
                    if (self.interrupt_detectron_event.is_set()):
                        self.interrupt_detectron_event.clear()
                        self.logger.info("Interrupt event detected, stopping detection")
                        break
                    elif (self.stopping):
                        self.logger.info(f"Stopping BallDetector due to stopping condition, thus ball detected")
                        break
                    

                    if self.cfg.detectron_visualize:
                        v = Visualizer(
                            color_image,
                            metadata=self.metadata,
                            scale=1.0,
                            instance_mode=None
                        )
                        out = v.draw_instance_predictions(instances)
                        result_bgr = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)
                        cv2.imshow("Detectron2 RealSense", result_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
        finally:
            self.logger.info("Shutting down BallDetector")
            self.pipeline.stop()
            cv2.destroyAllWindows()