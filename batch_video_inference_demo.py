from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import decord
import os
import random
import json

""" MAIN PACKAGES:
torch==1.10.1
torchvision==0.11.2
transformers==4.16.0
timm==0.4.12 
fairscale==0.4.4
decord==0.6.0
einops==0.6.0
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_video(video_file, num_frames, temporal_stride, image_size, random_sampling_start=False):
    """
    :param video_file: path to video
    :param num_frames: number of sampled frames from the video
    :param temporal_stride: temporal stride for sampling frames
    :param image_size: size of each frame
    :param random_sampling_start: whether randomly sampling the start frame index, default=False.
    :return: torch tensor with the shape of (T, C, H, W)
    """
    # read video using decord
    decord_vr = decord.VideoReader(video_file, num_threads=1)
    total_frames = len(decord_vr)

    # sampling frame indices
    span_frames = num_frames * temporal_stride
    if random_sampling_start and total_frames > span_frames:
        start_frame_idx = random.randint(0, total_frames - span_frames)
    else:
        start_frame_idx = 0
    frame_indices = []
    for i in range(num_frames):
        frame_idx = start_frame_idx + i * temporal_stride
        if frame_idx < total_frames:
            frame_indices.append(frame_idx)
        else:
            frame_indices = frame_indices + [total_frames - 1] * (num_frames - len(frame_indices))
            break

    # load frames
    frames = decord_vr.get_batch(frame_indices).asnumpy() # numpy array
    transform = transforms.Compose([
        transforms.Resize((image_size ,image_size) ,interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    video_data = []
    for i in range(num_frames):
        # convert to PIL Images
        frame = Image.fromarray(frames[i, :, :, :]).convert('RGB')
        # transform
        video_data.append(transform(frame))
    video_data = torch.stack(video_data, dim=0) # (T, C, H, W)

    return video_data




from models.blip import blip_decoder

# a batch of emotional video files
data_root = './batch_video_inference_demo/video' # use videos from the MAFW dataset (https://mafw-database.github.io/MAFW/)
video_files = [
    os.path.join(data_root, '00019.mp4'),
    os.path.join(data_root, '00020.mp4'),
    os.path.join(data_root, '00021.mp4'),
    os.path.join(data_root, '00022.mp4'),
]
# several parameters
image_size = 512
num_frames = 6
temporal_stride = 16
random_sampling_start=True
batch_videos = []
for video_file in video_files:
    video = load_video(video_file, num_frames, temporal_stride, image_size, random_sampling_start=random_sampling_start)
    batch_videos.append(video)
batch_videos = torch.stack(batch_videos, dim=0).to(device) # (B, T, C, H, W)


# set prompt and load model
# prompt = 'a picture of ' # default
prompt = "a video of "
# prompt = "an emotional video of "
# prompt = "Describe the emotional behavior: "
# prompt = "Describe the subject's emotional behavior: "
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base', prompt=prompt)
model.eval()
model = model.to(device)

# batch inference
temporal_poolings = [None, 'avg', 'max', 'min']
saved_dir = './batch_video_inference_demo/caption'
os.makedirs(saved_dir, exist_ok=True)
with torch.no_grad():
    for temporal_pooling in temporal_poolings:
        # beam search
        print("=" * 20 + f"Caption using beam search (temporal pooling: {temporal_pooling})" + "=" * 20)

        captions = model.generate(batch_videos, sample=False, num_beams=5, max_length=30, min_length=15, temporal_pooling=temporal_pooling)
        for i, video_file in enumerate(video_files):
            print(f"\t'{os.path.basename(video_file)}': {captions[i]}")
        # save to json file
        json_file = os.path.join(saved_dir, f'caption_using_beam_search_tem_pool_{str(temporal_pooling)}.json')
        with open(json_file, "w") as f:
            json_data = [{'video': v, 'caption': c}for v,c in (zip(video_files, captions))]
            print(json_data)
            json.dump(json_data, f)
            print(f"\tsave caption to '{json_file}'")

        # nucleus sampling
        print("=" * 20 + f"Caption using nucleus search (temporal pooling: {temporal_pooling})" + "=" * 20)
        captions = model.generate(batch_videos, sample=True, top_p=0.9, max_length=30, min_length=15, temporal_pooling=temporal_pooling)
        for i, video_file in enumerate(video_files):
            print(f"\t'{os.path.basename(video_file)}': {captions[i]}")
        # save to json file
        json_file = os.path.join(saved_dir, f'caption_using_nucleus_search_tem_pool_{str(temporal_pooling)}.json')
        with open(json_file, "w") as f:
            json_data = [{'video': v, 'caption': c}for v,c in (zip(video_files, captions))]
            print(json_data)
            json.dump(json_data, f)
            print(f"\tsave caption to '{json_file}'")


""" MODEL OUTPUT:
reshape position embedding from 196 to 1024
load checkpoint from https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth
====================Caption using beam search (temporal pooling: None)====================
	'00019.mp4': a man in a black shirt talking on a cell phone
	'00020.mp4': a man in a black shirt looking through a glass door
	'00021.mp4': a man in a white shirt and a woman in a white dress
	'00022.mp4': a woman with a headband talking to a man in a field
[{'video': './batch_video_inference_demo/video/00019.mp4', 'caption': 'a man in a black shirt talking on a cell phone'}, {'video': './batch_video_inference_demo/video/00020.mp4', 'caption': 'a man in a black shirt looking through a glass door'}, {'video': './batch_video_inference_demo/video/00021.mp4', 'caption': 'a man in a white shirt and a woman in a white dress'}, {'video': './batch_video_inference_demo/video/00022.mp4', 'caption': 'a woman with a headband talking to a man in a field'}]
	save caption to './batch_video_inference_demo/caption/caption_using_beam_search_tem_pool_None.json'
====================Caption using nucleus search (temporal pooling: None)====================
	'00019.mp4': robert harle's new movie character, < 3
	'00020.mp4': a man looking out at the camera with a capt saying i amedora
	'00021.mp4': the same guy in japan and the first is just that they did to make fun, i do nothing you need it
	'00022.mp4': a woman wearing a gray dress, and holding a book
[{'video': './batch_video_inference_demo/video/00019.mp4', 'caption': "robert harle's new movie character, < 3"}, {'video': './batch_video_inference_demo/video/00020.mp4', 'caption': 'a man looking out at the camera with a capt saying i amedora'}, {'video': './batch_video_inference_demo/video/00021.mp4', 'caption': 'the same guy in japan and the first is just that they did to make fun, i do nothing you need it'}, {'video': './batch_video_inference_demo/video/00022.mp4', 'caption': 'a woman wearing a gray dress, and holding a book'}]
	save caption to './batch_video_inference_demo/caption/caption_using_nucleus_search_tem_pool_None.json'
====================Caption using beam search (temporal pooling: avg)====================
	'00019.mp4': the vampire season 3 episode 3 - damon - damon - damon - damon - damon - damon - damon - damon - damon - damon
	'00020.mp4': a man in a scene with a man in the background
	'00021.mp4': a man in a bathroom with a woman in the background
	'00022.mp4': a woman in a field with a man in the background
[{'video': './batch_video_inference_demo/video/00019.mp4', 'caption': 'the vampire season 3 episode 3 - damon - damon - damon - damon - damon - damon - damon - damon - damon - damon'}, {'video': './batch_video_inference_demo/video/00020.mp4', 'caption': 'a man in a scene with a man in the background'}, {'video': './batch_video_inference_demo/video/00021.mp4', 'caption': 'a man in a bathroom with a woman in the background'}, {'video': './batch_video_inference_demo/video/00022.mp4', 'caption': 'a woman in a field with a man in the background'}]
	save caption to './batch_video_inference_demo/caption/caption_using_beam_search_tem_pool_avg.json'
====================Caption using nucleus search (temporal pooling: avg)====================
	'00019.mp4': tom jonas in supernatural news on youtube snjc1t
	'00020.mp4': ben michael thomas's with the arrow 2018 movie review trailer hd hd
	'00021.mp4': a man with the man shirtless in a bathroom in an image
	'00022.mp4': the movie adaptation, in this week trailer for'a woman's'to death & other '
[{'video': './batch_video_inference_demo/video/00019.mp4', 'caption': 'tom jonas in supernatural news on youtube snjc1t'}, {'video': './batch_video_inference_demo/video/00020.mp4', 'caption': "ben michael thomas's with the arrow 2018 movie review trailer hd hd"}, {'video': './batch_video_inference_demo/video/00021.mp4', 'caption': 'a man with the man shirtless in a bathroom in an image'}, {'video': './batch_video_inference_demo/video/00022.mp4', 'caption': "the movie adaptation, in this week trailer for'a woman's'to death & other '"}]
	save caption to './batch_video_inference_demo/caption/caption_using_nucleus_search_tem_pool_avg.json'
====================Caption using beam search (temporal pooling: max)====================
	'00019.mp4': a a a a a a a a a a a a a a a a a a a a a a a a a a
	'00020.mp4': g g g g g g g g g g g g g g g g g g g g g g g g g g
	'00021.mp4': n n n n n n n n n n n n n n n n n n n n n n n n n n
	'00022.mp4': w w w w w w w w w w w w w w w w w w w w w w w w w w
[{'video': './batch_video_inference_demo/video/00019.mp4', 'caption': 'a a a a a a a a a a a a a a a a a a a a a a a a a a'}, {'video': './batch_video_inference_demo/video/00020.mp4', 'caption': 'g g g g g g g g g g g g g g g g g g g g g g g g g g'}, {'video': './batch_video_inference_demo/video/00021.mp4', 'caption': 'n n n n n n n n n n n n n n n n n n n n n n n n n n'}, {'video': './batch_video_inference_demo/video/00022.mp4', 'caption': 'w w w w w w w w w w w w w w w w w w w w w w w w w w'}]
	save caption to './batch_video_inference_demo/caption/caption_using_beam_search_tem_pool_max.json'
====================Caption using nucleus search (temporal pooling: max)====================
	'00019.mp4':  left tu 12 &'+ super tu n g br q k'a g top & g with l u 2 s all
	'00020.mp4': g na, and n b right in g i o id left with g behind all app'two u with s u g behind
	'00021.mp4': and the or 1 & un tu the br a with tr'un i w and g can n 2 top br'g sc
	'00022.mp4': tu d on a s and'for in & or two w at cal two on l w g la m br a, i
[{'video': './batch_video_inference_demo/video/00019.mp4', 'caption': " left tu 12 &'+ super tu n g br q k'a g top & g with l u 2 s all"}, {'video': './batch_video_inference_demo/video/00020.mp4', 'caption': "g na, and n b right in g i o id left with g behind all app'two u with s u g behind"}, {'video': './batch_video_inference_demo/video/00021.mp4', 'caption': "and the or 1 & un tu the br a with tr'un i w and g can n 2 top br'g sc"}, {'video': './batch_video_inference_demo/video/00022.mp4', 'caption': "tu d on a s and'for in & or two w at cal two on l w g la m br a, i"}]
	save caption to './batch_video_inference_demo/caption/caption_using_nucleus_search_tem_pool_max.json'
====================Caption using beam search (temporal pooling: min)====================
	'00019.mp4': a man in a black shirt talking on a cell while he is wearing a black shirt
	'00020.mp4': the the the the the and the the the the and the the and the the and the the and the the and the and the
	'00021.mp4': a man in a white shirt standing in front of a woman in a white dress
	'00022.mp4': in the the the the the the the the the the the the the the the the the the the the the the the the the
[{'video': './batch_video_inference_demo/video/00019.mp4', 'caption': 'a man in a black shirt talking on a cell while he is wearing a black shirt'}, {'video': './batch_video_inference_demo/video/00020.mp4', 'caption': 'the the the the the and the the the the and the the and the the and the the and the the and the and the'}, {'video': './batch_video_inference_demo/video/00021.mp4', 'caption': 'a man in a white shirt standing in front of a woman in a white dress'}, {'video': './batch_video_inference_demo/video/00022.mp4', 'caption': 'in the the the the the the the the the the the the the the the the the the the the the the the the the'}]
	save caption to './batch_video_inference_demo/caption/caption_using_beam_search_tem_pool_min.json'
====================Caption using nucleus search (temporal pooling: min)====================
	'00019.mp4': daniel's the vampire movie shows some of his films - video
	'00020.mp4': fu art 1 american street e a'and art art term a fashion in hot music, secret - the education to shade online on
	'00021.mp4': an action that includes a very emotional, interesting, but the person is an official fact
	'00022.mp4': history one the and the short and term the street, all history secret secret - french in art free the new new smart long the
[{'video': './batch_video_inference_demo/video/00019.mp4', 'caption': "daniel's the vampire movie shows some of his films - video"}, {'video': './batch_video_inference_demo/video/00020.mp4', 'caption': "fu art 1 american street e a'and art art term a fashion in hot music, secret - the education to shade online on"}, {'video': './batch_video_inference_demo/video/00021.mp4', 'caption': 'an action that includes a very emotional, interesting, but the person is an official fact'}, {'video': './batch_video_inference_demo/video/00022.mp4', 'caption': 'history one the and the short and term the street, all history secret secret - french in art free the new new smart long the'}]
	save caption to './batch_video_inference_demo/caption/caption_using_nucleus_search_tem_pool_min.json'
"""