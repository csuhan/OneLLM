import sys
import os
import argparse
import multiprocessing as mp
import numpy as np
from typing import List, Optional

import torch
import torch.distributed as dist

from fairscale.nn.model_parallel import initialize as fs_init

import gradio as gr
from util.misc import setup_for_distributed
from util.misc import default_tensor_type
from model.meta import MetaModel
from data.conversation_lib import conv_templates, SeparatorStyle
from PIL import Image
import torchvision.transforms as transforms
from data.fintune_dataset import make_audio_features
from data import video_utils 
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
from data.fintune_dataset import pc_norm
from functools import partial
import glob
import torchvision.transforms.functional as F

T_random_resized_crop = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=3,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class PairRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias) for img in imgs]

class PairToTensor(transforms.ToTensor):
    def __call__(self, pics):
        return [F.to_tensor(pic) for pic in pics]

class PairNormalize(transforms.Normalize):
    def forward(self, tensors):
        return [F.normalize(tensor, self.mean, self.std, self.inplace) for tensor in tensors]
    
transform_pairimg_train = transforms.Compose([
    PairRandomResizedCrop(size=(224, 224), scale=(0.99, 1.0), ratio=(0.75, 1.3333), interpolation=3, antialias=None),  # 3 is bicubic
    PairToTensor(),
    PairNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None] #[1, 128, 1024]
    return fbank
    
def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_point(point_path):
    point_feat = np.load(point_path)
    point_feat = torch.tensor(point_feat)
    point_feat = pc_norm(point_feat)
    return point_feat

def load_fmri(fmri_path):
    data = np.load(fmri_path)
    data = data.mean(axis=0)
    data = torch.tensor(data[None])
    return data

def load_rgbx(image_path, x_image_path):
    # trick: replace path if 'depth_scaled' in path
    x_image_path = x_image_path.replace('depth_scaled', 'depth')

    image = Image.open(image_path).convert('RGB')
    x_image = Image.open(x_image_path).convert('RGB')
    x_image = x_image.resize(image.size[-2:])

    image, x_image = transform_pairimg_train([image, x_image])

    # [2, 3, H, W]
    image = torch.stack([image, x_image], dim=0)
    return image


class Ready: pass


def model_worker(
    rank: int, args: argparse.Namespace, barrier: mp.Barrier,
    request_queue: mp.Queue, response_queue: Optional[mp.Queue] = None,
) -> None:
    """
    The worker function that manipulates the GPU to run the inference.
    Exact n_gpu workers are started, with each one operating on a separate GPU.
    Args:
        rank (int): Distributed rank of the worker.
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    world_size = len(args.gpu_ids)
    gpu_id = args.gpu_ids[rank]
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    print(f"| distributed init on worker {rank}/{world_size}. "
          f"using gpu: {gpu_id}")
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(gpu_id)

    torch.manual_seed(1)
    np.random.seed(1)

    # set the print behavior.
    setup_for_distributed(rank == 0)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }[args.dtype]
    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel(args.llama_type, args.llama_config, tokenizer_path=args.tokenizer_path)
    for ckpt_id in range(args.num_ckpts):
        ckpt_path = hf_hub_download(repo_id=args.pretrained_path, filename=args.ckpt_format.format(str(ckpt_id)))
        # ckpt_path = os.path.join(args.pretrained_path, args.ckpt_format.format(str(ckpt_id)))
        print(f"Loading pretrained weights {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
    # print("load result:\n", msg)
    model.cuda()
    model.eval()
    print(f"Model = {str(model)}")

    barrier.wait()

    while True:
        if response_queue is not None:
            response_queue.put(Ready())
        img_path, audio_path, video_path, point_path, fmri_path, depth_path, depth_rgb_path, normal_path, normal_rgb_path, chatbot, max_gen_len, temperature, top_p, modality = request_queue.get()
        try:
            if 'image' in modality and img_path is not None:
                image = Image.open(img_path).convert('RGB')
                inputs = T_random_resized_crop(image)
            elif 'video' in modality and video_path is not None:
                inputs = load_video(video_path)
            elif 'audio' in modality and audio_path is not None:
                inputs = load_audio(audio_path)
            elif 'point' in modality and point_path is not None:
                inputs = load_point(point_path)
            elif 'fmri' in modality and fmri_path is not None:
                inputs = load_fmri(fmri_path)
            elif 'rgbd' in modality and depth_path is not None and depth_rgb_path is not None:
                inputs = load_rgbx(depth_rgb_path, depth_path)
            elif 'rgbn' in modality and normal_path is not None and normal_rgb_path is not None:
                inputs = load_rgbx(normal_rgb_path, normal_path)
            else:
                inputs = None
        except:
            inputs = None
        
        if inputs is not None:
            inputs = inputs[None].cuda().to(target_dtype)
    
        conv = conv_templates["v1"].copy()
        for user, bot in chatbot:
            conv.append_message(conv.roles[0], user)
            conv.append_message(conv.roles[1], bot)

        with torch.cuda.amp.autocast(dtype=target_dtype):
            print(conv.get_prompt())
            for stream_response in model.stream_generate(
                conv.get_prompt(), inputs,
                max_gen_len=max_gen_len, temperature=temperature, top_p=top_p,
                modal = modality
            ):
                conv_sep = (
                    conv.sep
                    if conv.sep_style == SeparatorStyle.SINGLE
                    else conv.sep2
                )
                end_pos = stream_response["text"].find(conv_sep)
                if end_pos != -1:
                    stream_response["text"] = (
                        stream_response['text'][:end_pos].rstrip() + "\n"
                    )
                    stream_response["end_of_content"] = True

                # keep a few characters if not end_of_content to avoid sending
                # part of conv_sep before all of it is generated.
                if not stream_response["end_of_content"]:
                    if len(stream_response["text"]) < len(conv_sep):
                        continue
                    stream_response["text"] = (
                        stream_response["text"][:-len(conv_sep)]
                    )

                if response_queue is not None:
                    response_queue.put(stream_response)

                if stream_response["end_of_content"]:
                    break


def gradio_worker(
    request_queues: List[mp.Queue], response_queue: mp.Queue,
    args: argparse.Namespace, barrier: mp.Barrier,
) -> None:
    """
    The gradio worker is responsible for displaying the WebUI and relay the
    requests to model workers. It should be launched only once.
    Args:
        request_queues (List[mp.Queue]): A list of request queues (one for
            each model worker).
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    def show_user_input(msg, chatbot):
        return "", chatbot + [[msg, None]]

    def stream_model_output(img_path, audio_path, video_path, point_path, fmri_path, depth_path, depth_rgb_path, normal_path, normal_rgb_path, chatbot, max_gen_len, gen_t, top_p, modality):
        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, Ready):
                break
        for queue in request_queues:
            queue.put((img_path, audio_path, video_path, point_path, fmri_path, depth_path, depth_rgb_path, normal_path, normal_rgb_path, chatbot, max_gen_len, gen_t, top_p, modality))
        while True:
            content_piece = response_queue.get()
            chatbot[-1][1] = content_piece["text"]
            yield chatbot
            if content_piece["end_of_content"]:
                break

    def undo(chatbot):
        if len(chatbot) > 0:
            chatbot = chatbot[:-1]
        return chatbot

    def clear():
        chatbot = []
        msg = ""
        return chatbot, msg
    
    def show_point_cloud(file):
        point = load_point(file).numpy()
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=point[:,0], y=point[:,1], z=point[:,2],
                    mode='markers',
                    marker=dict(
                    size=1.2,
                    color=['rgb({},{},{})'.format(r, g, b) for r,g,b in zip(point[:,3], point[:,4], point[:,5])]
                ))],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)
                )),)
        return fig
    
    def change_modality(modal):
        return modal

    CSS ="""
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; overflow: auto;}
    """

    header="""
    ## OneLLM: One Framework to Align All Modalities with Language
    [[Project Page](https://onellm.csuhan.com)] [[Paper](https://arxiv.org/abs/2312.03700)] [[Code](https://github.com/csuhan/OneLLM)]
    """

    with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
        gr.Markdown(header)
        with gr.Row(equal_height=True):
            modality = gr.Textbox(value='image', visible=False)
            with gr.Column(scale=1):
                with gr.Tab('Image') as img_tab:
                    img_path = gr.Image(label='Image Input', type='filepath')
                    # gr.Examples(
                    #     examples=[
                    #         "examples/new_york.jpg",
                    #         "examples/food_menu.png",
                    #         ],
                    #     inputs=[img_path],
                    # )
                with gr.Tab('Video') as video_tab:
                    video_path = gr.Video(label='Video Input', max_length=180)

                with gr.Tab('Audio') as audio_tab:
                    audio_path = gr.Audio(label='Audio Input', type='filepath', sources=['upload'])

                with gr.Tab('Point Cloud') as point_tab:
                    point_path = gr.File(label='Point Cloud Input', elem_id="pointpath", elem_classes="")
                    point_vis = gr.Plot()
                    btn = gr.Button(value="Show Point Cloud")
                    btn.click(show_point_cloud, point_path, point_vis)

                with gr.Tab('IMU') as imu_tab:
                    gr.Markdown('Coming soon🤗')
                with gr.Tab('fMRI') as fmri_tab:
                    fmri_path = gr.File(label='fMRI Input', elem_id="fmripath", elem_classes="")
                    fmri_image_path = gr.Image(label='Reference Image', interactive=False)

                with gr.Tab('Depth Map') as depth_tab:
                    depth_path = gr.Image(label='Depth Map', type='filepath')
                    depth_rgb_path = gr.Image(label='RGB Image', type='filepath')

                with gr.Tab('Normal Map') as normal_tab:
                    normal_path = gr.Image(label='Normal Map', type='filepath')
                    normal_rgb_path = gr.Image(label='RGB Image', type='filepath')

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(elem_id="chatbot")
                msg = gr.Textbox()

                with gr.Row():
                    submit_button = gr.Button("Submit", variant="primary")
                    undo_button = gr.Button("Undo")
                    clear_button = gr.ClearButton([chatbot, msg, img_path, audio_path, video_path, point_path, fmri_path, depth_path, depth_rgb_path, normal_path, normal_rgb_path, point_vis])
                with gr.Row():
                    max_gen_len = gr.Slider(
                        minimum=1, maximum=args.model_max_seq_len // 2,
                        value=args.model_max_seq_len // 2, interactive=True,
                        label="Single-turn max response length",
                    )
                    gen_t = gr.Slider(
                        minimum=0, maximum=1, value=0.1, interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0, maximum=1, value=0.75, interactive=True,
                        label="Top-p",
                    )
        
        img_tab.select(partial(change_modality, 'image'), [], [modality])
        video_tab.select(partial(change_modality, 'video'), [], [modality])
        audio_tab.select(partial(change_modality, 'audio'), [], [modality])
        point_tab.select(partial(change_modality, 'point'), [], [modality])
        fmri_tab.select(partial(change_modality, 'fmri'), [], [modality])
        depth_tab.select(partial(change_modality, 'rgbd'), [], [modality])
        normal_tab.select(partial(change_modality, 'rgbn'), [], [modality])

        img_path.change(clear, [], [chatbot, msg])
        audio_path.change(clear, [], [chatbot, msg])
        video_path.change(clear, [], [chatbot, msg])
        point_path.change(clear, [], [chatbot, msg])
        fmri_path.change(clear, [], [chatbot, msg])
        depth_path.change(clear, [], [chatbot, msg])
        normal_path.change(clear, [], [chatbot, msg])

        msg.submit(
            show_user_input, [msg, chatbot], [msg, chatbot],
        ).then(
            stream_model_output, [img_path, audio_path, video_path, point_path, fmri_path, depth_path, depth_rgb_path, normal_path, normal_rgb_path, chatbot, max_gen_len, gen_t, top_p, modality], chatbot,
        )
        submit_button.click(
            show_user_input, [msg, chatbot], [msg, chatbot],
        ).then(
            stream_model_output, [img_path, audio_path, video_path, point_path, fmri_path, depth_path, depth_rgb_path, normal_path, normal_rgb_path, chatbot, max_gen_len, gen_t, top_p, modality], chatbot,
        )
        undo_button.click(undo, chatbot, chatbot)
    barrier.wait()
    demo.queue(api_open=True).launch(share=True, max_threads=1)


@dataclass
class DemoConfig:
    gpu_ids = [0]
    tokenizer_path = "config/llama2/tokenizer.model"
    llama_type = "onellm"
    llama_config = "config/llama2/7B.json"
    model_max_seq_len = 2048
    pretrained_path = "csuhan/OneLLM-7B-hf"
    # pretrained_path = "/home/pgao/jiaming/weights/7B_v20_splits/"
    ckpt_format = "consolidated.00-of-01.s{}.pth"
    num_ckpts = 10
    master_port = 23863
    master_addr = "127.0.0.1"
    dtype = "fp16"

if __name__ == "__main__":
    args = DemoConfig()

    # using the default "fork" method messes up some imported libs (e.g.,
    # pandas)
    # mp.set_start_method("spawn")

    # setup the queues and start the model workers
    request_queues = []
    response_queue = mp.Queue()
    worker_processes = []
    barrier = mp.Barrier(len(args.gpu_ids) + 1)
    for rank, gpu_id in enumerate(args.gpu_ids):
        request_queue = mp.Queue()
        rank_response_queue = response_queue if rank == 0 else None
        process = mp.Process(
            target=model_worker,
            args=(rank, args, barrier, request_queue, rank_response_queue),
        )
        process.start()
        worker_processes.append(process)
        request_queues.append(request_queue)

    gradio_worker(request_queues, response_queue, args, barrier)