import json
import os
import random
import warnings
import traceback
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
from decord import VideoReader, cpu
from dataset.dataset_util import euler2rotm, rotm2euler
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange
from dataset.video_transforms import Resize_Preprocess, ToTensorVideo
from util import update_paths


class Dataset_3D(Dataset):
    def __init__(
            self,
            args, mode='val'
    ):
        """Constructor: Initialize dataset based on mode (train/val/test)."""
        super().__init__()
        self.args = args
        self.mode = mode
        self.sequence_length = args.num_frames  # Total frames per sample (e.g., 17 for 16 predicted frames)
        self.max_stride = args.get('max_stride', 5)  # Max random stride for train mode
        self.cam_ids = args.cam_ids
        self.accumulate_action = args.accumulate_action

        # 动态配置action_dim和scaler
        self.dual_arm = getattr(args, 'dual_arm', False)  # 默认单臂模式

        if not self.dual_arm:
            # Action configuration (xyz:3, rpy:3, gripper:1)
            self.action_dim = 7
            self.c_act_scaler = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=float)
        else:
            # Action configuration for dual-arm: left_arm(xyz:3, rpy:3) + right_arm(xyz:3, rpy:3) + left_gripper:1 + right_gripper:1
            self.action_dim = 14  # 6+6+1+1 = 14 dimensions
            # self.c_act_scaler = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0,  # left arm scaling
            #                             20.0, 20.0, 20.0, 20.0, 20.0, 20.0,  # right arm scaling
            #                             1.0, 1.0], dtype=float)  # left and right gripper scaling
            self.c_act_scaler = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0,  # left arm scaling
                                          20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=float)  # right arm scaling

        # Load annotation paths
        if mode == 'train':
            self.data_path = args.train_annotation_path
        elif mode == 'val':
            self.data_path = args.val_annotation_path
            self.start_frame_interval = args.val_start_frame_interval  # Fixed for val
            self.sequence_interval = args.sequence_interval          # Fixed for val
        elif mode == 'test':
            self.data_path = args.test_annotation_path
            self.start_frame_interval = args.val_start_frame_interval  # Fixed for test
            self.sequence_interval = args.sequence_interval          # Fixed for test
        self.video_path = args.video_path
        self.ann_files = self._init_anns(self.data_path)

        # Initialize samples/episodes (train = dynamic, val/test = fixed)
        if mode == 'train':
            self.episodes = self._init_episodes(self.ann_files)  # List of (ann_file, total_frames)
            self.episodes = [ep for ep in self.episodes if ep[1] >= self.sequence_length]  # Filter short episodes
            self.samples = self.episodes
            self.num_samples = len(self.episodes)
            print(f"Train Mode: {len(self.episodes)} valid episodes (≥ {self.sequence_length} frames)")
        else:
            self.samples = self._init_fixed_samples(self.ann_files)  # Precomputed fixed samples
            self.samples = sorted(self.samples, key=lambda x: (x['ann_file'], x['frame_ids'][0]))
            if args.debug and not args.do_evaluate:
                self.samples = self.samples[:10]  # Debug: limit samples
            self.num_samples = len(self.samples)
            print(f"{mode} Mode: {len(self.ann_files)} trajectories, {len(self.samples)} fixed samples")

        # Image transformations
        self.preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size)),  # e.g., (288, 512)
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.not_norm_preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size))
        ])

        self.wrong_number = 0  # Track invalid samples (for error handling)

    def __str__(self):
        return f"{self.num_samples} samples from {self.data_path} (mode: {self.mode})"

    def _init_anns(self, data_dir):
        """Load all annotation files (*.json) from the data directory."""
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]

    # --------------------------------------------------------------------------
    # Train Mode: Initialize episodes (no precomputed frames)
    # --------------------------------------------------------------------------
    def _init_episodes(self, ann_files):
        """For train mode: Get (ann_file, total_frames) for each valid episode."""
        episodes = []
        with ThreadPoolExecutor(32) as executor:
            futures = {executor.submit(self._get_episode_frames, ann): ann for ann in ann_files}
            for future in tqdm(as_completed(futures), total=len(ann_files), desc="Loading train episodes"):
                result = future.result()
                if result is not None:
                    episodes.append(result)
        return episodes

    def _get_episode_frames(self, ann_file):
        """Get total frames for a single annotation file (handle errors)."""
        try:
            with open(ann_file, "r") as f:
                ann = json.load(f)
            if 'state' in ann:
                return (ann_file, len(ann['state']))  # Frames = len(state)
            else:
                return (ann_file, len(ann.get('action', [])) + 1)  # Frames = len(action) + 1 (delta actions)
        except Exception as e:
            warnings.warn(f"Skipping invalid ann file: {ann_file} (error: {str(e)})")
            return None

    # --------------------------------------------------------------------------
    # Val/Test Mode: Precompute fixed samples (original logic)
    # --------------------------------------------------------------------------
    def _init_fixed_samples(self, ann_files):
        """For val/test: Precompute fixed samples with fixed start/stride."""
        samples = []
        with ThreadPoolExecutor(32) as executor:
            futures = {executor.submit(self._process_fixed_ann, ann): ann for ann in ann_files}
            for future in tqdm(as_completed(futures), total=len(ann_files), desc=f"Loading {self.mode} samples"):
                samples.extend(future.result())
        return samples

    def _process_fixed_ann(self, ann_file):
        """Generate fixed frame_ids for a single annotation file (val/test)."""
        samples = []
        try:
            with open(ann_file, "r") as f:
                ann = json.load(f)
        except Exception as e:
            warnings.warn(f"Skipping ann file: {ann_file} (error: {str(e)})")
            return samples

        # Get total frames
        if 'state' in ann:
            n_frames = len(ann['state'])
        else:
            n_frames = len(ann.get('action', [])) + 1

        # Generate fixed frame_ids
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = {'ann_file': ann_file, 'frame_ids': []}
            curr_frame = frame_i
            while len(sample['frame_ids']) < self.sequence_length and curr_frame < n_frames:
                sample['frame_ids'].append(curr_frame)
                curr_frame += self.sequence_interval
            if len(sample['frame_ids']) == self.sequence_length:
                samples.append(sample)
        return samples

    # --------------------------------------------------------------------------
    # Video/Frame Loading Utilities
    # --------------------------------------------------------------------------
    def _load_video(self, video_path, frame_ids):
        """Load raw video frames using Decord."""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all(), f"Frame IDs exceed video length ({len(vr)})"
        assert (np.array(frame_ids) >= 0).all(), "Negative frame IDs"
        return vr.get_batch(frame_ids).asnumpy()  # (L, H, W, 3)

    def _load_tokenized_video(self, video_path, frame_ids):
        """Load pre-encoded (tokenized) video frames."""
        video_tensor = torch.load(video_path)
        assert (np.array(frame_ids) < video_tensor.size(0)).all(), "Frame IDs exceed tokenized video length"
        return video_tensor[frame_ids]  # (L, C, H, W)

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        """Load video/latent and apply transformations."""
        # Select camera (random if not specified)
        if cam_id is None:
            cam_id = random.choice(self.cam_ids)

        # Load video/latent
        if pre_encode:
            video_path = os.path.join(self.video_path, label['latent_videos'][cam_id]['latent_video_path'])
            frames = self._load_tokenized_video(video_path, frame_ids)
        else:
            video_path = os.path.join(self.video_path, label['videos'][cam_id]['video_path'])
            frames = self._load_video(video_path, frame_ids).astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (L, C, H, W)
            # Apply preprocessing
            if self.args.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, cam_id

    # --------------------------------------------------------------------------
    # Action Loading/Accumulation Utilities
    # --------------------------------------------------------------------------
    def _get_robot_states(self, label, frame_ids):
        """Load absolute robot states (arm + gripper) at given frame_ids."""
        if not self.dual_arm:
            all_states = np.array(label['state'], dtype=np.float32)
            all_gripper = np.array(label['continuous_gripper_state'], dtype=np.float32)
            states = all_states[frame_ids]
            gripper = all_gripper[frame_ids]
            # assert states.shape == (self.sequence_length, 7), "Invalid arm state shape"
            # assert gripper.shape == (self.sequence_length,), "Invalid gripper state shape"
            return states[:, :6], gripper  # (L, 6) arm states, (L,) gripper
        else:
            """Load absolute robot states (dual-arm + grippers) at given frame_ids."""
            all_states = np.array(label['state'], dtype=np.float32)  # Shape: (T, 2, 7)
            all_gripper = np.array(label['continuous_gripper_state'], dtype=np.float32)  # Shape: (T, 2)
            
            states = all_states[frame_ids]  # Shape: (L, 2, 7)
            gripper = all_gripper[frame_ids]  # Shape: (L, 2)
            
            # Extract left and right arm states (6D each)
            left_arm_states = states[:, 0, :6]   # (L, 6) - left arm xyz+rpy
            right_arm_states = states[:, 1, :6]  # (L, 6) - right arm xyz+rpy
            left_gripper = gripper[:, 0]         # (L,) - left gripper
            right_gripper = gripper[:, 1]        # (L,) - right gripper
            
            # Combine into dual-arm format
            dual_arm_states = np.concatenate([left_arm_states, right_arm_states], axis=1)  # (L, 12)
            dual_gripper_states = np.stack([left_gripper, right_gripper], axis=1)  # (L, 2)
            
            # assert dual_arm_states.shape == (self.sequence_length, 12), f"Invalid dual-arm state shape: {dual_arm_states.shape}"
            # assert dual_gripper_states.shape == (self.sequence_length, 2), f"Invalid dual-gripper state shape: {dual_gripper_states.shape}"
            
            return dual_arm_states, dual_gripper_states

    def _get_all_robot_states(self, label, frame_ids):
        if not self.dual_arm:
            all_states = np.array(label['state'])
            all_cont_gripper_states = np.array(label['continuous_gripper_state'])

            states = all_states[frame_ids]
            cont_gripper_states = all_cont_gripper_states[frame_ids]
            
            arm_states = states[:, :6]  # 只取前6维 (x,y,z,rx,ry,rz)
            return arm_states, cont_gripper_states
        else:
            all_states = np.array(label['state'])
            all_cont_gripper_states = np.array(label['continuous_gripper_state'])

            states = all_states[frame_ids]
            cont_gripper_states = all_cont_gripper_states[frame_ids]
            
            # Extract left and right arm states (6D each)
            left_arm_states = states[:, 0, :6]   # (L, 6) - left arm xyz+rpy
            right_arm_states = states[:, 1, :6]  # (L, 6) - right arm xyz+rpy
            left_gripper = cont_gripper_states[:, 0]         # (L,) - left gripper
            right_gripper = cont_gripper_states[:, 1]        # (L,) - right gripper

            # Combine into dual-arm format
            dual_arm_states = np.concatenate([left_arm_states, right_arm_states], axis=1)  # (L, 12)
            dual_gripper_states = np.stack([left_gripper, right_gripper], axis=1)  # (L, 2)

            return dual_arm_states, dual_gripper_states

    def _get_actions_from_states(self, arm_states, gripper_states):
        if not self.dual_arm:
            """Compute actions from absolute states (automatically accumulates over stride)."""
            actions = np.zeros((self.sequence_length - 1, self.action_dim), dtype=np.float32)
            if self.accumulate_action:
                # Accumulate relative to the first frame
                first_rotm = euler2rotm(arm_states[0, 3:6])
                first_xyz = arm_states[0, :3]
                for k in range(1, self.sequence_length):
                    curr_xyz = arm_states[k, :3]
                    curr_rotm = euler2rotm(arm_states[k, 3:6])
                    # Relative xyz/rpy
                    rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                    rel_rpy = rotm2euler(first_rotm.T @ curr_rotm)
                    # Assign action
                    actions[k-1, :3] = rel_xyz
                    actions[k-1, 3:6] = rel_rpy
                    actions[k-1, 6] = gripper_states[k]
            else:
                # Relative to previous frame (still accumulates over stride)
                for k in range(1, self.sequence_length):
                    prev_xyz = arm_states[k-1, :3]
                    prev_rotm = euler2rotm(arm_states[k-1, 3:6])
                    curr_xyz = arm_states[k, :3]
                    curr_rotm = euler2rotm(arm_states[k, 3:6])
                    # Relative xyz/rpy
                    rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                    rel_rpy = rotm2euler(prev_rotm.T @ curr_rotm)
                    # Assign action
                    actions[k-1, :3] = rel_xyz
                    actions[k-1, 3:6] = rel_rpy
                    actions[k-1, 6] = gripper_states[k]
            return torch.from_numpy(actions)
        else:
            """Compute actions from absolute states for dual-arm (automatically accumulates over stride)."""
            actions = np.zeros((self.sequence_length - 1, self.action_dim), dtype=np.float32)
            
            if self.accumulate_action:
                # Accumulate relative to the first frame for both arms
                # Left arm
                left_first_rotm = euler2rotm(arm_states[0, 3:6])
                left_first_xyz = arm_states[0, :3]
                # Right arm  
                right_first_rotm = euler2rotm(arm_states[0, 9:12])
                right_first_xyz = arm_states[0, 6:9]
                
                for k in range(1, self.sequence_length):
                    # Left arm relative motion
                    left_curr_xyz = arm_states[k, :3]
                    left_curr_rotm = euler2rotm(arm_states[k, 3:6])
                    left_rel_xyz = np.dot(left_first_rotm.T, left_curr_xyz - left_first_xyz)
                    left_rel_rpy = rotm2euler(left_first_rotm.T @ left_curr_rotm)
                    
                    # Right arm relative motion
                    right_curr_xyz = arm_states[k, 6:9]
                    right_curr_rotm = euler2rotm(arm_states[k, 9:12])
                    right_rel_xyz = np.dot(right_first_rotm.T, right_curr_xyz - right_first_xyz)
                    right_rel_rpy = rotm2euler(right_first_rotm.T @ right_curr_rotm)

                    # Assign actions: (6+1)+(6+1)
                    actions[k-1, :3] = left_rel_xyz
                    actions[k-1, 3:6] = left_rel_rpy
                    actions[k-1, 6] = gripper_states[k, 0]  # left gripper
                    actions[k-1, 7:10] = right_rel_xyz
                    actions[k-1, 10:13] = right_rel_rpy
                    actions[k-1, 13] = gripper_states[k, 1]  # right gripper
            else:
                # Relative to previous frame for both arms
                for k in range(1, self.sequence_length):
                    # Left arm
                    left_prev_xyz = arm_states[k-1, :3]
                    left_prev_rotm = euler2rotm(arm_states[k-1, 3:6])
                    left_curr_xyz = arm_states[k, :3]
                    left_curr_rotm = euler2rotm(arm_states[k, 3:6])
                    left_rel_xyz = np.dot(left_prev_rotm.T, left_curr_xyz - left_prev_xyz)
                    left_rel_rpy = rotm2euler(left_prev_rotm.T @ left_curr_rotm)
                    
                    # Right arm
                    right_prev_xyz = arm_states[k-1, 6:9]
                    right_prev_rotm = euler2rotm(arm_states[k-1, 9:12])
                    right_curr_xyz = arm_states[k, 6:9]
                    right_curr_rotm = euler2rotm(arm_states[k, 9:12])
                    right_rel_xyz = np.dot(right_prev_rotm.T, right_curr_xyz - right_prev_xyz)
                    right_rel_rpy = rotm2euler(right_prev_rotm.T @ right_curr_rotm)

                    # Assign actions: (6+1)+(6+1)
                    actions[k-1, :3] = left_rel_xyz
                    actions[k-1, 3:6] = left_rel_rpy
                    actions[k-1, 6] = gripper_states[k, 0]  # left gripper
                    actions[k-1, 7:10] = right_rel_xyz
                    actions[k-1, 10:13] = right_rel_rpy
                    actions[k-1, 13] = gripper_states[k, 1]  # right gripper
                    
            return torch.from_numpy(actions)

    def _get_all_actions(self, arm_states, gripper_states, accumulate_action):
        action_num = arm_states.shape[0]-1
        if not self.dual_arm:
            action = np.zeros((action_num, self.action_dim))
            if accumulate_action:
                first_xyz = arm_states[0, 0:3]
                first_rpy = arm_states[0, 3:6]
                first_rotm = euler2rotm(first_rpy)
                for k in range(1, action_num+1):
                    curr_xyz = arm_states[k, 0:3]
                    curr_rpy = arm_states[k, 3:6]
                    curr_gripper = gripper_states[k]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                    rel_rotm = first_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, 0:3] = rel_xyz
                    action[k - 1, 3:6] = rel_rpy
                    action[k - 1, 6] = curr_gripper
            else:
                for k in range(1, action_num+1):
                    prev_xyz = arm_states[k - 1, 0:3]
                    prev_rpy = arm_states[k - 1, 3:6]
                    prev_rotm = euler2rotm(prev_rpy)
                    curr_xyz = arm_states[k, 0:3]
                    curr_rpy = arm_states[k, 3:6]
                    curr_gripper = gripper_states[k]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                    rel_rotm = prev_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, 0:3] = rel_xyz
                    action[k - 1, 3:6] = rel_rpy
                    action[k - 1, 6] = curr_gripper
            return torch.from_numpy(action)  # (l - 1, act_dim)
        else:
            action = np.zeros((action_num, self.action_dim))
            if accumulate_action:
                left_first_xyz = arm_states[0, 0:3]
                left_first_rpy = arm_states[0, 3:6]
                left_first_rotm = euler2rotm(left_first_rpy)

                right_first_xyz = arm_states[0, 6:9]
                right_first_rpy = arm_states[0, 9:12]
                right_first_rotm = euler2rotm(right_first_rpy)

                for k in range(1, action_num+1):
                    left_curr_xyz = arm_states[k, 0:3]
                    left_curr_rpy = arm_states[k, 3:6]
                    left_curr_rotm = euler2rotm(left_curr_rpy)
                    left_rel_xyz = np.dot(left_first_rotm.T, left_curr_xyz - left_first_xyz)
                    left_rel_rotm = left_first_rotm.T @ left_curr_rotm
                    left_rel_rpy = rotm2euler(left_rel_rotm)

                    right_curr_xyz = arm_states[k, 6:9]
                    right_curr_rpy = arm_states[k, 9:12]
                    right_curr_rotm = euler2rotm(right_curr_rpy)
                    right_rel_xyz = np.dot(right_first_rotm.T, right_curr_xyz - right_first_xyz)
                    right_rel_rotm = right_first_rotm.T @ right_curr_rotm
                    right_rel_rpy = rotm2euler(right_rel_rotm)

                    # Assign actions: (6+1)+(6+1)
                    action[k-1, :3] = left_rel_xyz
                    action[k-1, 3:6] = left_rel_rpy
                    action[k-1, 6] = gripper_states[k, 0]  # left gripper
                    action[k-1, 7:10] = right_rel_xyz
                    action[k-1, 10:13] = right_rel_rpy
                    action[k-1, 13] = gripper_states[k, 1]  # right gripper
            else:
                for k in range(1, action_num+1):
                    left_prev_xyz = arm_states[k - 1, 0:3]
                    left_prev_rpy = arm_states[k - 1, 3:6]
                    left_prev_rotm = euler2rotm(left_prev_rpy)

                    right_prev_xyz = arm_states[k - 1, 6:9]
                    right_prev_rpy = arm_states[k - 1, 9:12]
                    right_prev_rotm = euler2rotm(right_prev_rpy)

                    left_curr_xyz = arm_states[k, 0:3]
                    left_curr_rpy = arm_states[k, 3:6]
                    left_curr_rotm = euler2rotm(left_curr_rpy)
                    left_rel_xyz = np.dot(left_prev_rotm.T, left_curr_xyz - left_prev_xyz)
                    left_rel_rotm = left_prev_rotm.T @ left_curr_rotm
                    left_rel_rpy = rotm2euler(left_rel_rotm)

                    right_curr_xyz = arm_states[k, 6:9]
                    right_curr_rpy = arm_states[k, 9:12]
                    right_curr_rotm = euler2rotm(right_curr_rpy)
                    right_rel_xyz = np.dot(right_prev_rotm.T, right_curr_xyz - right_prev_xyz)
                    right_rel_rotm = right_prev_rotm.T @ right_curr_rotm
                    right_rel_rpy = rotm2euler(right_rel_rotm)

                    # Assign actions: (6+1)+(6+1)
                    action[k-1, :3] = left_rel_xyz
                    action[k-1, 3:6] = left_rel_rpy
                    action[k-1, 6] = gripper_states[k, 0]  # left gripper
                    action[k-1, 7:10] = right_rel_xyz
                    action[k-1, 10:13] = right_rel_rpy
                    action[k-1, 13] = gripper_states[k, 1]  # right gripper
            
            return torch.from_numpy(action)  # (l - 1, act_dim)

    def _accumulate_delta_actions(self, all_actions, frame_ids):
        if not self.dual_arm:
            """Accumulate precomputed delta actions over the stride (for no 'state' case)."""
            actions = []
            for i in range(len(frame_ids) - 1):
                f_prev = frame_ids[i]
                f_curr = frame_ids[i+1]
                # Slice actions: [f_prev, f_curr-1] (sum delta actions between f_prev and f_curr)
                act_slice = all_actions[f_prev:f_curr]  # end index is exclusive (f_curr-1 is last)
                # Accumulate xyz/rpy (sum over stride)
                accumulated_xyz_rpy = act_slice[:, :6].sum(axis=0)
                # Gripper: take last action's state (no sum for discrete gripper)
                accumulated_gripper = act_slice[-1, 6:7]
                # Combine into one action
                actions.append(np.concatenate([accumulated_xyz_rpy, accumulated_gripper]))
            return torch.from_numpy(np.array(actions, dtype=np.float32))
        else:
            """Accumulate precomputed delta actions over the stride for dual-arm (for no 'state' case)."""
            actions = []
            for i in range(len(frame_ids) - 1):
                f_prev = frame_ids[i]
                f_curr = frame_ids[i+1]
                # Slice actions: [f_prev, f_curr-1] (sum delta actions between f_prev and f_curr)
                act_slice = all_actions[f_prev:f_curr]  # end index is exclusive (f_curr-1 is last)
                
                # Accumulate xyz/rpy for both arms (sum over stride)
                accumulated_left_xyz_rpy = act_slice[:, :6].sum(axis=0)  # left arm
                accumulated_right_xyz_rpy = act_slice[:, 6:12].sum(axis=0)  # right arm
                
                # Gripper: take last action's state (no sum for discrete gripper)
                accumulated_left_gripper = act_slice[-1, 12:13]  # left gripper
                accumulated_right_gripper = act_slice[-1, 13:14]  # right gripper

                # Combine into one action: (6+1)+(6+1)
                actions.append(np.concatenate([
                    accumulated_left_xyz_rpy, 
                    accumulated_left_gripper, 
                    accumulated_right_xyz_rpy,
                    accumulated_right_gripper
                ]))
            return torch.from_numpy(np.array(actions, dtype=np.float32))

    # --------------------------------------------------------------------------
    # Main __getitem__ (dynamic for train, fixed for val/test)
    # --------------------------------------------------------------------------
    def __getitem__(self, index, cam_id=None, return_video=False):
        try:
            # --------------------------
            # Step 1: Get frame_ids
            # --------------------------
            if self.mode == 'train':
                # Train: Randomly sample start_frame and stride
                ann_file, n_frames = self.episodes[index]
                L = self.sequence_length

                # 1.1 Compute valid stride range (s)
                if L <= 1:
                    raise ValueError("sequence_length must be ≥ 2 for train mode")
                s_max = (n_frames - 1) // (L - 1)  # Max s to fit L frames
                s_max = min(s_max, self.max_stride)  # Clamp with user-defined max
                if s_max < 1:
                    raise ValueError(f"Episode too short: {n_frames} frames < {L} required")

                # 1.2 Randomly sample stride and start_frame
                sequence_stride = np.random.randint(1, s_max + 1)
                start_max = (n_frames - 1) - (L - 1) * sequence_stride  # Max valid start frame
                start_frame = np.random.randint(0, start_max + 1)

                # 1.3 Generate frame_ids
                frame_ids = [start_frame + i * sequence_stride for i in range(L)]
                assert max(frame_ids) < n_frames, "Frame IDs exceed episode length"

            else:
                # Val/Test: Use precomputed frame_ids
                sample = self.samples[index]
                ann_file = sample['ann_file']
                frame_ids = sample['frame_ids']
                # Verify frame validity
                with open(ann_file, "r") as f:
                    ann = json.load(f)
                n_frames = len(ann['state']) if 'state' in ann else len(ann.get('action', [])) + 1
                assert max(frame_ids) < n_frames, "Precomputed frame IDs invalid"

            # --------------------------
            # Step 2: Load annotations
            # --------------------------
            with open(ann_file, "r") as f:
                label = json.load(f)

            # --------------------------
            # Step 3: Compute actions (with accumulation for stride)
            # --------------------------
            if 'state' in label:
                # Case 1: Actions from absolute states (auto-accumulate over stride)
                arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                actions = self._get_actions_from_states(arm_states, gripper_states)
            else:
                # Case 2: Accumulate precomputed delta actions over stride
                all_actions = np.array(label['action'], dtype=np.float32)
                if len(all_actions) != n_frames - 1:
                    raise ValueError(f"Action count {len(all_actions)} != frames-1 {n_frames-1}")
                actions = self._accumulate_delta_actions(all_actions, frame_ids)

            # Apply action scaling
            actions *= self.c_act_scaler
            actions = actions.float()

            # --------------------------
            # Step 4: Load video/latent
            # --------------------------
            data = {'action': actions}
            if self.args.pre_encode:
                latent, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=True)
                data['latent'] = latent.float()
                if return_video:
                    video, _ = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                    data['video'] = video.float()
            else:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                data['video'] = video.float()

            # Add metadata
            data['video_name'] = {
                'episode_id': label['episode_id'],
                'start_frame_id': str(frame_ids[0]),
                'cam_id': str(cam_id),
                'stride': str(sequence_stride) if self.mode == 'train' else str(self.sequence_interval)
            }

            return data

        except Exception as e:
            # Handle invalid samples: retry with a random valid index
            error_msg = f"Error in __getitem__ (index {index}): {str(e)}\n{traceback.format_exc()}"
            warnings.warn(error_msg)
            self.wrong_number += 1
            if self.wrong_number > 100:
                raise RuntimeError("Exceeded 100 invalid samples. Check dataset integrity!")
            return self[np.random.randint(self.num_samples)]

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    # Example usage (update config paths as needed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/bridge/frame_ada.yaml")
    args = parser.parse_args()

    # Merge configs (base + experiment)
    data_config = OmegaConf.load("configs/base/data.yaml")
    diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    exp_config = OmegaConf.load(args.config)
    args = OmegaConf.merge(data_config, diffusion_config, exp_config)
    update_paths(args)

    # Add required train mode args (if missing in config)
    # args.max_stride = 5  # Max random stride for training
    args.max_stride = 5  # Max random stride for training
    args.num_frames = 16  # Example: 1 initial frame + 16 predicted frames

    # Test train dataset
    train_dataset = Dataset_3D(args, mode='train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,  # Shuffle train samples per epoch
        num_workers=4,
        pin_memory=True
    )

    # Test val dataset
    val_dataset = Dataset_3D(args, mode='val')
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,  # No shuffle for val
        num_workers=4,
        pin_memory=True
    )

    # Verify data shapes
    print("\n=== Train Loader Sample ===")
    for batch in train_loader:
        print(f"Video shape: {batch['video'].shape}")  # (B, L, C, H, W)
        print(f"Action shape: {batch['action'].shape}")  # (B, L-1, 7)
        print(f"Stride used: {batch['video_name']['stride'][:5]}")  # Show first 5 strides
        break

    print("\n=== Val Loader Sample ===")
    for batch in val_loader:
        print(f"Video shape: {batch['video'].shape}")
        print(f"Action shape: {batch['action'].shape}")
        print(f"Fixed stride: {batch['video_name']['stride'][0]}")  # Fixed stride
        break