import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames, frame_size):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.load_video(video_path)
        return frames, label

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) < self.num_frames:
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))

        frames = torch.stack(frames, dim=1)
        return frames
