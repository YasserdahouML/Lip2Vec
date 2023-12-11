from pathlib import Path

import torch
import torch.utils.data
import glob
import numpy as np
import torch.nn.functional as F
import os
import glob
import numpy as np
import sys

import cv2
import pickle
from .preprocess import *
from .transform import *
from .utils_data import *
import json


class LRS3(object):
    def __init__(self, data_dir,
        data_suffix='.mp4', 
        mean_face_path="20words_mean_face.npy",
        noise_file_path="babbleNoise_resample_16K.npy", 
        convert_gray=True,
        start_idx=48,
        stop_idx=68,
        ):

        self._data_dir = data_dir
        self._data_suffix = data_suffix
        self._window_margin = 12
        self.fps = 25
        self._convert_gray = convert_gray
        self._start_idx = start_idx
        self._stop_idx = stop_idx
        self._crop_width = 96
        self._crop_height = 96
        self.file = open('datasets/char_list.json')
        self.char_list = json.load(self.file)['char_list']
        self.special_chars = ['–','…','—','`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}',':',';','"','<',',','>','.','?','/']

        self._reference = np.load(
            os.path.join( os.path.dirname(__file__), mean_face_path)
        )
        self._noise = np.load(
            os.path.join( os.path.dirname(__file__), noise_file_path)
        )
        self.vid_transform = self.get_video_transform()
        self._data_files = []

        self.lrs3_labels = 'LRS3_test_nonumeric.ref'

        self.load_dataset()


    def load_dataset(self):

        # -- add examples to self._data_files
        self._get_files_for_partition()


        # -- from self._data_files to self.list
        self.list = dict()
        self.instance_ids = dict()
        for i, x in enumerate(self._data_files):
            #label = read_txt_lines( x[:-3] +'txt' )[0].split(':  ')[1]
            self.list[i] =  x
            self.instance_ids[i] = self._get_instance_id_from_path( x )

        print(f"Partition Testloaded")

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        instance_id = x.split('/')[-1]
        return os.path.splitext( instance_id )[0]

    def _get_label_from_path(self, x):
        return x.split('/')

    def _get_files_for_partition(self):
        # get rgb/mfcc file paths

        dir_fp = self._data_dir
        if not dir_fp:
            return

        search_str_original = os.path.join(dir_fp , '*', '*.mp4')
        self._data_files.extend( glob.glob( search_str_original ) )
        idx = [id for id,n in enumerate(self._data_files) if n.find('cropped') >= 0 or n.find('normalized') >= 0 ]
        self._data_files = np.delete(self._data_files, idx)

        print(len(self._data_files))


    def load_video(self, data_filename, landmarks_filename=None):
        """load_video.

        :param data_filename: str, the filename of input sequence.
        :param landmarks_filename: str, the filename of landmarks.
        """
        assert landmarks_filename is not None
        sequence, _ = self.preprocess(
            video_pathname=data_filename,
            landmarks_pathname=landmarks_filename,
        )
        return sequence
    

        
    def get_video_transform(self):
        """get_video_transform.

        :param speed_rate: float, the speed rate between the frame rate of \
            the input video and the frame rate used for training.
        """

        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
    
        return Compose([
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std),])
            
        

    def preprocess(self, video_pathname, landmarks_pathname):
        """preprocess.

        :param video_pathname: str, the filename for the video.
        :param landmarks_pathname: str, the filename for the landmarks.
        """
        # -- Step 1, extract landmarks from pkl files.
        if landmarks_pathname.endswith('pkl'):
            if isinstance(landmarks_pathname, str):
                with open(landmarks_pathname, "rb") as pkl_file:
                    landmarks = pickle.load(pkl_file)
        elif landmarks_pathname.endswith('npy'):
            landmarks = np.load(landmarks_pathname)
            landmarks = [i for i in landmarks]

        else:
            landmarks = landmarks_pathname
        # -- Step 2, pre-process landmarks: interpolate frames that not being detected.
        preprocessed_landmarks = self.landmarks_interpolate(landmarks)
        # -- Step 3, exclude corner cases:
        #   -- 1) no landmark in all frames
        #   -- 2) number of frames is less than window length.
        if not preprocessed_landmarks or len(preprocessed_landmarks) < self._window_margin: return
        # -- Step 4, affine transformation and crop patch 
        sequence, transformed_frame, transformed_landmarks = \
            self.crop_patch(video_pathname, preprocessed_landmarks)
        assert sequence is not None, "cannot crop from {}.".format(video_pathname)
        return sequence, preprocessed_landmarks


    def landmarks_interpolate(self, landmarks):
        """landmarks_interpolate.

        :param landmarks: List, the raw landmark (in-place)

        """
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        if not valid_frames_idx:
            return None
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
                continue
            else:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        # -- Corner case: keep frames at the beginning or at the end failed to be detected.
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
        return landmarks


    def crop_patch(self, video_pathname, landmarks):
        """crop_patch.

        :param video_pathname: str, the filename for the processed video.
        :param landmarks: List, the interpolated landmarks.
        """
        frame_idx = 0
        frame_gen = load_video(video_pathname)
        while True:
            try:
                frame = frame_gen.__next__() ## -- BGR
            except StopIteration:
                break
            if frame_idx == 0:
                sequence = []
                sequence_frame = []
                sequence_landmarks = []
            window_margin = min(self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean([landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            

            stable_points=(28, 33, 36, 39, 42, 45, 48, 54)

            transformed_frame, transformed_landmarks = self.affine_transform(
                frame,
                smoothed_landmarks,
                self._reference,
                grayscale=self._convert_gray,
                stable_points = stable_points,
            )
            sequence.append( cut_patch( transformed_frame,
                                        transformed_landmarks[self._start_idx:self._stop_idx],
                                        self._crop_height//2,
                                        self._crop_width//2,))

            sequence_frame.append( transformed_frame)
            sequence_landmarks.append( transformed_landmarks)
            frame_idx += 1


        return np.array(sequence), np.array(sequence_frame), np.array(sequence_landmarks)


    def get_video(self, video_pathname):
        """crop_patch.

        :param video_pathname: str, the filename for the processed video.
        :param landmarks: List, the interpolated landmarks.
        """
        frame_idx = 0
        frame_gen = load_video(video_pathname)
        while True:
            try:
                frame = frame_gen.__next__() ## -- BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except StopIteration:
                break
            if frame_idx == 0:
                sequence = []

            sequence.append( frame)
            frame_idx += 1
            
        return np.array(sequence)
    
    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=False,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0
    ):
        """affine_transform.

        :param frame: numpy.array, the input sequence.
        :param landmarks: List, the tracked landmarks.
        :param reference: numpy.array, the neutral reference frame.
        :param grayscale: bool, save as grayscale if set as True.
        :param target_size: tuple, size of the output image.
        :param reference_size: tuple, size of the neural reference frame.
        :param stable_points: tuple, landmark idx for the stable points.
        :param interpolation: interpolation method to be used.
        :param border_mode: Pixel extrapolation method .
        :param border_value: Value used in case of a constant border. By default, it is 0.
        """
        # Prepare everything
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if len(stable_points) < 10:
            return frame, landmarks
        
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

        # Warp the face patch and the landmarks
        transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                                stable_reference, method=cv2.LMEDS)[0]
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

        return transformed_frame, transformed_landmarks
    
 
    
    def read_txt_lines(self, filepath):
        assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
        with open( filepath ) as myfile:
            content = myfile.read().splitlines()
        return content

    def __getitem__(self, idx):    
        
        path_video = self.list[idx]
        
        path_landmark = self.list[idx][:-4] +'.pkl'
        path_label = self.list[idx][:-4] +'.txt'
        
        raw_data = self.load_video(path_video, path_landmark)
        preprocess_data = torch.FloatTensor(raw_data)
        preprocess_data = self.vid_transform(preprocess_data)

        label =  self.read_txt_lines( path_label )[0][7:]
        
        return preprocess_data, label                                                                                                                                                                                                                                               

    
    def __len__(self):
        return len(self._data_files)




def build(reps_set, args):
    root = Path(args.lrs3_path)
    assert root.exists(), f'provided LRS3 path {root} does not exist'
    PATHS = {
        "test": (root / "test"),
    }

    video_folder = PATHS[reps_set]

    dataset = LRS3(data_dir=video_folder, data_suffix='.mp4')
    
    return dataset