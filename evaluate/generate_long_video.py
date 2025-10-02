# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import torch
import imageio
import argparse
import shutil
import torchvision.transforms as T
from tqdm import tqdm
from einops import rearrange, repeat
from datetime import datetime
from diffusers.models import AutoencoderKL

from models import get_models
from dataset import get_dataset
from evaluate.generate_short_video import generate_single_video
from util import get_args, update_paths



def main(args,rank,thread,thread_num):
    device = torch.device("cuda", rank)
    args.latent_size = [t // 8 for t in args.video_size]
    model = get_models(args)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)

    train_steps = int(args.evaluate_checkpoint.split('/')[-1][0:-3])
    current_date = datetime.now()
    experiment_dir = f"{args.results_dir}/{current_date.strftime('%m')}/{current_date.strftime('%d')}/{args.anno}-debug"
    episode_latent_videos_dir = f'{experiment_dir}/checkpoints/{train_steps:07d}/{args.mode}_episode_latent_videos'
    episode_videos_dir = f'{experiment_dir}/checkpoints/{train_steps:07d}/{args.mode}_episode_videos'

    os.makedirs(episode_latent_videos_dir, exist_ok=True)
    os.makedirs(episode_videos_dir, exist_ok=True)

    if args.evaluate_checkpoint:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=lambda storage, loc: storage, weights_only=False)
        if "ema" in checkpoint:
            print('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                print('Ignoring: {}'.format(k))
        print('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Successfully load model at {}!'.format(args.evaluate_checkpoint)) 
    model.to(device)
    model.eval()

    train_dataset,val_dataset = get_dataset(args)

    def printvideo(videos,filename):
        t_videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
        t_videos = rearrange(t_videos, 'f c h w -> f h w c')
        t_videos = t_videos.numpy()
        writer = imageio.get_writer(filename, fps=20) 
        # writer = imageio.get_writer(filename, fps=15)
        for frame in t_videos:
            writer.append_data(frame) 
        writer.close()

    for sample_idx, ann_file in tqdm(enumerate(val_dataset.ann_files),total=len(val_dataset.ann_files)):
        if sample_idx % thread_num == thread:
            with open(ann_file, "rb") as f:
                ann = json.load(f)
            episode_id = ann['episode_id']
            ann_id = ann_file.split('/')[-1].split('.')[0]
            if args.dataset == 'languagetable':
                output_video_path = os.path.join(episode_videos_dir,f'{episode_id}.mp4')
                output_latents_path = os.path.join(episode_latent_videos_dir,f'{episode_id}.pt')
            else:
                output_video_path = os.path.join(episode_videos_dir,f'{ann_id}.mp4')
                output_latents_path = os.path.join(episode_latent_videos_dir,f'{ann_id}.pt')

            # if os.path.exists(output_latents_path) and os.path.exists(output_latents_path):
            #     continue
            
            if args.dataset == 'languagetable':
                video_path = os.path.join(args.video_path, ann['video_path'])
                latent_video_path = os.path.join(args.video_path, ann['latent_video_path'])
                with open(latent_video_path, 'rb') as file:
                    latent_video = torch.load(file)['obs']
            else:
                video_path = os.path.join(args.video_path, ann['videos'][0]['video_path'])
            
            video_reader = imageio.get_reader(video_path)
            video_frames = []
            for frame in video_reader:
                frame_tensor = torch.tensor(frame).permute(2, 0, 1)  # HWC -> CHW
                video_frames.append(frame_tensor)
            video_reader.close()
            video_tensor = torch.stack(video_frames).unsqueeze(0)  # (1, F, C, H, W)
            
            # 使用VAE编码，参考短视频脚本的逻辑
            video_tensor = video_tensor.to(device).float() / 255.0 * 2.0 - 1.0  # 归一化到[-1,1]
            # 添加resize操作，确保编码后的潜在表示尺寸是32x40
            # VAE的缩放因子是8，所以需要resize到256x320
            resize_transform = T.Resize((256, 320), antialias=True)
            b, f, c, h, w = video_tensor.shape
            # 对每一帧进行resize
            resized_frames = []
            for i in range(f):
                resized_frame = resize_transform(video_tensor[0, i])  # 处理单帧
                resized_frames.append(resized_frame)
            video_tensor = torch.stack(resized_frames).unsqueeze(0)  # 重新组装
            
            b, f, c, h, w = video_tensor.shape
            video_flat = rearrange(video_tensor, 'b f c h w -> (b f) c h w').contiguous()
            
            # 分批编码以避免内存问题
            stride = getattr(args, 'local_val_batch_size', 8)
            encode_video_list = []
            for i in range(0, video_flat.size(0), stride):
                batch_frames = video_flat[i:i+stride].to(device)
                with torch.no_grad():  # 避免梯度计算
                    encoded = vae.encode(batch_frames).latent_dist.sample().mul_(vae.config.scaling_factor)
                encode_video_list.append(encoded.cpu()) # 立即移到CPU释放GPU内存
            
            encoded_video = torch.cat(encode_video_list, dim=0)
            latent_video = rearrange(encoded_video, '(b f) c h w -> b f c h w', b=b, f=f).squeeze(0)

            if args.model == 'VDM':
                latent_video = video_tensor
                latent_video = latent_video.permute(0, 3, 1, 2)
                latent_video = val_dataset.preprocess(latent_video)

            # total_frame = latent_video.size()[0]
            
            # 限制frame_ids不超过状态数据的长度
            # max_available_frames = len(ann['state'])
            if 'state' in ann:
                max_available_frames = len(ann['state'])
            else:
                # If no state, use action count (actions are deltas: frames = actions + 1)
                max_available_frames = len(ann.get('action', [])) + 1
                print(max_available_frames)
            
            # total_frame = min(total_frame, max_available_frames)
            # use json state/action length to generate long video
            total_frame = max_available_frames
            
            frame_ids = list(range(total_frame))
            if args.dataset == 'languagetable':
                action = torch.tensor(ann['actions'])
            else:
                # if 'state' in ann:
                #     print("Use json state in generate_long_video!")
                #     arm_states, gripper_states = val_dataset._get_all_robot_states(ann, frame_ids)
                #     action = val_dataset._get_all_actions(arm_states, gripper_states, args.accumulate_action)
                #     print("The first arm state: ", arm_states[0])
                #     # action = val_dataset._get_actions_from_states(arm_states, gripper_states)
                # else:
                print("Use json action directly in generate_long_video!")
                # action = torch.tensor(ann['action'])
                all_action_raw = torch.tensor(ann['action']) # (Bs, 2, 7)
                action_raw = all_action_raw[frame_ids[:-1]]
                Bs = action_raw.shape[0]
                action = torch.zeros(Bs, 14)
                action[:, :7] = action_raw[:, 0, :7]
                action[:, 7:14] = action_raw[:, 1, :7]
                    # 修正夹爪状态：从 continuous_gripper_state 中获取第7维
                    # if 'continuous_gripper_state' in ann:
                    #     continuous_gripper_states = torch.tensor(ann['continuous_gripper_state'])
                    #     # 获取对应帧的夹爪状态
                    #     #gripper_states_for_frames = continuous_gripper_states[frame_ids]
                    #     gripper_states_for_frames = continuous_gripper_states[frame_ids[:-1]] #TODO: Kevin fix
                    #     # 更新 action 的第7维（索引6）为正确的夹爪状态
                    #     action[:, 6] = gripper_states_for_frames[:]  # 跳过第一帧，因为 action 比 state 少一帧

            action = action*val_dataset.c_act_scaler


            current_frame = 0
            start_image = latent_video[current_frame]
            seg_video_list = []
            seg_idx = 0
            
            latent_list = [latent_video[0:1].to(device)]

            while current_frame+args.num_frames-1 < total_frame:
                seg_action = action[current_frame:current_frame+args.num_frames-1]
                start_image = start_image.unsqueeze(0).unsqueeze(0)
                seg_action = seg_action.unsqueeze(0)
                seg_video, seg_latents = generate_single_video(args, start_image, seg_action, device, vae, model)
                seg_video = seg_video.squeeze()
                
                if args.model == 'VDM':
                    start_image = seg_video[-1].clone()
                else:
                    seg_latents = seg_latents.squeeze()
                    start_image = seg_latents[-1].clone()
                    latent_list.append(seg_latents[1:])

                current_frame += args.num_frames-1
                seg_video_list.append(seg_video[1:])
                seg_idx += 1
            
            seg_action = action[current_frame:]
            true_action = seg_action.size()[0]
            if true_action != 0:
                false_action = args.num_frames-true_action-1
                seg_false_action = repeat(seg_action[0], 'd -> f d', f= false_action) 
                
                com_action = torch.cat([seg_action,seg_false_action],dim=0)
                start_image = start_image.unsqueeze(0).unsqueeze(0)
                com_action = com_action.unsqueeze(0)
                seg_video, seg_latents = generate_single_video(args, start_image, com_action, device, vae, model)
                seg_video = seg_video.squeeze()
                seg_video_list.append(seg_video[1:true_action+1])
                if args.model != 'VDM':
                    seg_latents = seg_latents.squeeze()
                    latent_list.append(seg_latents[1:true_action+1])


            com_video = torch.cat(seg_video_list,dim=0).cpu()                 
            printvideo(com_video, output_video_path)
            print(output_video_path)

            if args.model != 'VDM':
                latents = torch.cat(latent_list,dim=0).cpu()    
                with open(output_latents_path,'wb') as file:
                    torch.save(latents,file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/rt1/frame_ada.yaml")
    parser.add_argument("--data_config", type=str, default="data_droid.yaml")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--thread", type=int, default=0)
    parser.add_argument("--thread-num", type=int, default=1)
    args = parser.parse_args()
    rank = args.rank
    thread = args.thread
    thread_num = args.thread_num
    args = get_args(args)
    update_paths(args)
    main(args,rank, thread, thread_num)