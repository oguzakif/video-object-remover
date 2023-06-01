from PIL import Image
import gradio as gr
from FGT_codes.tool.video_inpainting import video_inpainting
from SiamMask.tools.test import *
from SiamMask.experiments.siammask_sharp.custom import Custom
from types import SimpleNamespace
import torch
import numpy as np
import torchvision
import cv2
import sys
from os.path import exists, join, basename, splitext
import os
import argparse
from datetime import datetime

project_name = ''

sys.path.append(project_name)

sys.path.append(os.path.abspath(join(project_name, 'FGT_codes')))
sys.path.append(os.path.abspath(join(project_name, 'FGT_codes', 'tool')))
sys.path.append(os.path.abspath(join(project_name, 'FGT_codes', 'tool','configs')))
sys.path.append(os.path.abspath(join(project_name, 'FGT_codes', 'LAFC', 'flowCheckPoint')))
sys.path.append(os.path.abspath(join(project_name, 'FGT_codes', 'LAFC', 'checkpoint')))
sys.path.append(os.path.abspath(join(project_name, 'FGT_codes', 'FGT', 'checkpoint')))
sys.path.append(os.path.abspath(join(project_name, 'FGT_codes', 'LAFC',
                'flowCheckPoint', 'raft-things.pth')))

exp_path = join(project_name, 'SiamMask/experiments/siammask_sharp')
pretrained_path1 = join(exp_path, 'SiamMask_DAVIS.pth')

print(sys.path)

torch.set_grad_enabled(False)

# init SiamMask
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = load_config(SimpleNamespace(config=join(exp_path, 'config_davis.json')))
siammask = Custom(anchors=cfg['anchors'])
siammask = load_pretrain(siammask, pretrained_path1)
siammask = siammask.eval().to(device)

parser = argparse.ArgumentParser()
# parser.add_argument('--opt', default='configs/object_removal.yaml',
#                     help='Please select your config file for inference')
parser.add_argument('--opt', default=os.path.abspath(join(project_name, 'FGT_codes', 'tool','configs','object_removal.yaml')),
                    help='Please select your config file for inference')
# video completion
parser.add_argument('--mode', default='object_removal', choices=[
    'object_removal', 'watermark_removal', 'video_extrapolation'], help="modes: object_removal / video_extrapolation")
parser.add_argument(
    '--path', default='/myData/davis_resized/walking', help="dataset for evaluation")
parser.add_argument(
    '--path_mask', default='/myData/dilateAnnotations_4/walking', help="mask for object removal")
parser.add_argument(
    '--outroot', default=os.path.abspath(project_name), help="output directory")
parser.add_argument(
    '--outfilename', default="result.mp4", help="output filename")
parser.add_argument('--consistencyThres', dest='consistencyThres', default=5, type=float,
                    help='flow consistency error threshold')
parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
parser.add_argument('--Nonlocal', dest='Nonlocal',
                    default=False, type=bool)

# RAFT
# parser.add_argument(
#     '--raft_model', default='../LAFC/flowCheckPoint/raft-things.pth', help="restore checkpoint")
parser.add_argument(
    '--raft_model', default=os.path.abspath(join(project_name, 'FGT_codes', 'LAFC','flowCheckPoint','raft-things.pth')), help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision',
                    action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true',
                    help='use efficent correlation implementation')

# LAFC
# parser.add_argument('--lafc_ckpts', type=str, default='../LAFC/checkpoint')
parser.add_argument('--lafc_ckpts', type=str, default=os.path.abspath(join(project_name, 'FGT_codes', 'LAFC','checkpoint')))

# FGT
# parser.add_argument('--fgt_ckpts', type=str, default='../FGT/checkpoint')
parser.add_argument('--fgt_ckpts', type=str, default=os.path.abspath(join(project_name, 'FGT_codes', 'FGT','checkpoint')))


# extrapolation
parser.add_argument('--H_scale', dest='H_scale', default=2,
                    type=float, help='H extrapolation scale')
parser.add_argument('--W_scale', dest='W_scale', default=2,
                    type=float, help='W extrapolation scale')

# Image basic information
parser.add_argument('--imgH', type=int, default=256)
parser.add_argument('--imgW', type=int, default=432)
parser.add_argument('--flow_mask_dilates', type=int, default=8)
parser.add_argument('--frame_dilates', type=int, default=0)

parser.add_argument('--gpu', type=int, default=0)

# FGT inference parameters
parser.add_argument('--step', type=int, default=10)
parser.add_argument('--num_ref', type=int, default=-1)
parser.add_argument('--neighbor_stride', type=int, default=5)

parser.add_argument('--out_fps', type=int, default=24)

# visualization
parser.add_argument('--vis_flows', action='store_true',
                    help='Visualize the initialized flows')
parser.add_argument('--vis_completed_flows',
                    action='store_true', help='Visualize the completed flows')
parser.add_argument('--vis_prop', action='store_true',
                    help='Visualize the frames after stage-I filling (flow guided content propagation)')
parser.add_argument('--vis_frame', action='store_true',
                    help='Visualize frames')

args = parser.parse_args()


def getBoundaries(mask):
    if mask is None:
        return 0, 0, 0, 0

    indexes = np.where((mask == [255, 255, 255]).all(axis=2))
    print(indexes)
    x1 = min(indexes[1])
    y1 = min(indexes[0])
    x2 = max(indexes[1])
    y2 = max(indexes[0])

    return x1, y1, (x2-x1), (y2-y1)


def track_and_mask(vid, masked_frame, original_list, mask_list, in_fps, dt_string):
    x, y, w, h = getBoundaries(masked_frame)
    f = 0
    
    #turn 3d mask into 2d mask
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    #add first mask frame of the video by default
    mask_list.append(masked_frame)
    video_capture = cv2.VideoCapture()

    if video_capture.open(vid):
        width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        in_fps = fps
        # can't write out mp4, so try to write into an AVI file
        video_writer = cv2.VideoWriter(
            dt_string+"_output.avi", cv2.VideoWriter_fourcc(*'MP42'), fps, (width, height))

        while video_capture.isOpened():
            ret, frame = video_capture.read()

            if not ret:
                break
            
            # frame = cv2.resize(frame, (w - w % 8, h - h % 8))
            if f == 0:
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                # init tracker
                state = siamese_init(
                    frame, target_pos, target_sz, siammask, cfg['hp'], device=device)
                original_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame[:, :, 2] = (masked_frame > 0) * 255 + \
                        (masked_frame == 0) * frame[:, :, 2]
            else:
                # track
                state = siamese_track(
                    state, frame, mask_enable=True, refine_enable=True, device=device)
                original_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mask = state['mask'] > state['p'].seg_thr
                frame[:, :, 2] = (mask > 0) * 255 + \
                        (mask == 0) * frame[:, :, 2]

                mask = mask.astype(np.uint8)  # convert to an unsigned byte
                mask = mask * 255
                mask_list.append(mask)

            video_writer.write(frame)

            f = f + 1

        video_capture.release()
        video_writer.release()

    else:
        print("can't open the given input video file!")

    outname = (dt_string+"_output.avi")
    print('Original Frame Count: ',len(original_list))
    print('Mask Frame Count: ',len(mask_list))
    return original_list, mask_list, in_fps, outname


def inpaint_video(original_frame_list, mask_list, in_fps, dt_string):
    outname = (dt_string+"_result.mp4")
    args.out_fps = in_fps
    args.outfilename = outname
    video_inpainting(args, original_frame_list, mask_list)
    original_frame_list = []
    mask_list = []
    return outname,original_frame_list, mask_list


def get_first_frame(video):
    if(video == None):
        return gr.ImageMask()
    video_capture = cv2.VideoCapture()
    if video_capture.open(video):
        width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_capture.isOpened():
        ret, frame = video_capture.read()
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return RGB_frame


def drawRectangle(frame, mask):
    x1, y1, x2, y2 = getBoundaries(mask)

    return cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


def getStartEndPoints(mask):
    if mask is None:
        return 0, 0, 0, 0

    indexes = np.where((mask == [255, 255, 255]).all(axis=2))
    print(indexes)
    x1 = min(indexes[1])
    y1 = min(indexes[0])
    x2 = max(indexes[1])
    y2 = max(indexes[0])

    return x1, y1, x2, y2

def reset_components():
    return gr.update(value=None),gr.update(value=None, interactive=False),gr.update(value=None, interactive=False), [],[],24,datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

title = """<h1 align="center">Video Object Remover</h1>"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(
    """
    - Start uploading the video you wanted to edit.
    - Select the object you want to remove from the video.
    - Click on Run to start the process.
    """)
    original_frame_list = gr.State([])
    mask_list = gr.State([])
    # constants
    in_fps = gr.State(24)
    dt_string = gr.State(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                in_video = gr.PlayableVideo(label="Input Video", show_progress=True)
            with gr.Row():
                first_frame = gr.ImageMask(label="Select Object")
            with gr.Row():
                approve_mask = gr.Button(value="Run",variant="primary")
        with gr.Column(scale=1):
            with gr.Row():
                original_image = gr.Image(interactive=False)
            with gr.Row():
                masked_image = gr.Image(interactive=False)
        with gr.Column(scale=2):
            out_video = gr.Video(label="Segmented Video", show_progress=True)
            out_video_inpaint = gr.Video(label="Inpainted Video", show_progress=True)
            # track_mask = gr.Button(value="Track and Mask")
            # inpaint = gr.Button(value="Inpaint")

    in_video.change(fn=get_first_frame, inputs=[
                    in_video], outputs=[first_frame])
    in_video.clear(fn=reset_components, outputs=[first_frame, original_image, masked_image, original_frame_list, mask_list, in_fps, dt_string])
    approve_mask.click(lambda x: [x['image'], x['mask']], first_frame, [
                       original_image, masked_image])
    masked_image.change(fn=track_and_mask,inputs=[
                     in_video, masked_image, original_frame_list, mask_list, in_fps, dt_string], outputs=[original_frame_list, mask_list, in_fps, out_video])
    out_video.change(fn=inpaint_video, inputs=[original_frame_list, mask_list, in_fps, dt_string], outputs=[out_video_inpaint, original_frame_list, mask_list])
    # track_mask.click(fn=track_and_mask, inputs=[
    #                  in_video, masked_image, original_frame_list, mask_list, in_fps, dt_string], outputs=[original_frame_list, mask_list, in_fps, out_video])
    # inpaint.click(fn=inpaint_video, inputs=[original_frame_list, mask_list, in_fps, dt_string],
    #               outputs=[out_video_inpaint, original_frame_list, mask_list])


demo.launch(debug=True)
