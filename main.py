import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils import visualizer
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from PIL import Image
import numpy as np
import cv2

# set title
st.title('Pinned insect specimen detection')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = './model/model.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # This is required
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

# load image
if file:
    im = Image.open(file)
    im = np.asarray(im)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], scale=0.6)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(out.get_image()[:, :, ::-1])
    
