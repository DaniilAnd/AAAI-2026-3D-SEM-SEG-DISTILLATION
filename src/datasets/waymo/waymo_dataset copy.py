import os
from functools import lru_cache
from typing import Optional

import numpy as np

from src.datasets.dataset import Dataset
from src.datasets.frame_patcher import FramePatcher
from src.datasets.waymo.waymo_frame_patcher import WaymoFramePatcher
from src.datasets.waymo.waymo_scene_iterator import WaymoSceneIterator
from src.datasets.waymo.waymo_utils import find_all_scenes, load_scene_descriptor, get_frame_point_cloud, \
    get_instance_point_cloud, get_frame_index


class WaymoDataset(Dataset):
    def __init__(self,
                 dataset_root: str):
        self.__dataset_root = dataset_root
        self.__scene_ids = find_all_scenes(dataset_root=dataset_root)

    @property
    def dataroot(self) -> str:
        return self.__dataset_root

    @property
    def scenes(self) -> list:
        return list(self.__scene_ids)

    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)
        return WaymoSceneIterator(scene_id=scene_id, scene_descriptor=scene_descriptor)

    def load_frame_patcher(self,
                           scene_id: str,
                           frame_id: str) -> FramePatcher:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)

        return WaymoFramePatcher.load(dataset_root=self.__dataset_root,
                                      scene_id=scene_id,
                                      frame_id=frame_id,
                                      scene_descriptor=scene_descriptor)

    def can_serialise_frame_point_cloud(self,
                                        scene_id: str,
                                        frame_id: str) -> bool:
        path_to_save = self.__get_patched_frame_path(scene_id=scene_id,
                                                     frame_id=frame_id)

        # We can serialise point cloud if there is no point cloud saved.
        return not os.path.exists(path_to_save)

    def serialise_frame_point_clouds(self,
                                     scene_id: str,
                                     frame_id: str,
                                     frame_point_cloud: np.ndarray) -> Optional[str]:
        path_to_save = self.__get_patched_frame_path(scene_id=scene_id,
                                                     frame_id=frame_id)

        WaymoFramePatcher.serialise(path=path_to_save,
                                    point_cloud=frame_point_cloud)

        return path_to_save

    def get_frame_point_cloud(self,
                              scene_id: str,
                              frame_id: str) -> np.ndarray:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)
        frame_descriptor = scene_descriptor[frame_id]

        return get_frame_point_cloud(dataset_root=self.__dataset_root,
                                     scene_id=scene_id,
                                     frame_descriptor=frame_descriptor)

    def get_instance_point_cloud(self,
                                 scene_id: str,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)
        return get_instance_point_cloud(frame_point_cloud=frame_point_cloud,
                                        instance_id=instance_id,
                                        frame_descriptor=scene_descriptor[frame_id])

    @lru_cache(maxsize=12)
    def __load_scene_descriptor(self,
                                scene_id: str) -> dict:
        return load_scene_descriptor(dataset_root=self.__dataset_root,
                                     scene_id=scene_id)

    def __get_patched_frame_path(self,
                                 scene_id: str,
                                 frame_id: str):
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)
        frame_descriptor = scene_descriptor[frame_id]

        patched_root_folder = os.path.join(self.__dataset_root, 'patched')
        os.makedirs(patched_root_folder, exist_ok=True)

        patched_scene_folder = os.path.join(patched_root_folder, scene_id)
        os.makedirs(patched_scene_folder, exist_ok=True)

        frame_index = get_frame_index(frame_descriptor)
        return os.path.join(patched_scene_folder, f"{frame_index:04d}.npy")



from __future__ import annotations

import numpy as np

from src.datasets.frame_patcher import FramePatcher
from src.datasets.waymo.waymo_utils import get_frame_point_cloud, reapply_frame_transformation
from src.utils.geometry_utils import points_in_box


class WaymoFramePatcher(FramePatcher):
    """Patches NuScenes frames with new point clouds.
    """

    def __init__(self,
                 scene_id: str,
                 frame_id: str,
                 frame_point_cloud: np.ndarray,
                 frame_descriptor: dict):
        self.__scene_id = scene_id
        self.__frame_id = frame_id
        self.__frame_point_cloud = frame_point_cloud
        self.__frame_descriptor = frame_descriptor

    @classmethod
    def load(cls,
             dataset_root: str,
             scene_id: str,
             frame_id: str,
             scene_descriptor: dict) -> WaymoFramePatcher:
        lidar_point_cloud = get_frame_point_cloud(dataset_root=dataset_root,
                                                  scene_id=scene_id,
                                                  frame_descriptor=scene_descriptor[frame_id])

        return WaymoFramePatcher(scene_id=scene_id,
                                 frame_id=frame_id,
                                 frame_point_cloud=lidar_point_cloud,
                                 frame_descriptor=scene_descriptor[frame_id])

    @classmethod
    def serialise(cls,
                  path: str,
                  point_cloud: np.ndarray):
        """Serialises the given frame into a .npy file.
        """

        if not path.endswith('.npy'):
            raise Exception(f"Supports only npy files, got: {path}")

        np.save(path, point_cloud.T)

    @property
    def scene_id(self) -> str:
        return self.__scene_id

    @property
    def frame_id(self) -> str:
        return self.__frame_id

    @property
    def frame(self) -> np.ndarray:
        return self.__frame_point_cloud

    def patch_instance(self,
                       instance_id: str,
                       point_cloud: np.ndarray):
        annotations = self.__frame_descriptor['annos']
        ids = annotations['obj_ids']

        instance_index = np.where(ids == instance_id)
        instance_column = instance_index[0][0]

        center_xyz = annotations['location'][instance_column, :]
        dimensions_lwh = annotations['dimensions'][instance_column, :]
        heading_angle = annotations['heading_angles'][instance_column]

        points = self.__frame_point_cloud[0:3, :]
        mask = points_in_box(center_xyz=center_xyz,
                             dimensions_lwh=dimensions_lwh,
                             heading_angle=heading_angle,
                             points=points)

        # Remove masked elements in frame.
        self.__frame_point_cloud = self.__frame_point_cloud[:, np.where(~mask)[0]]

        # Put the object back into the scene.
        point_cloud = reapply_frame_transformation(point_cloud=point_cloud,
                                                   instance_id=instance_id,
                                                   frame_descriptor=self.__frame_descriptor)

        # Append instance patch: append should happen along
        self.__frame_point_cloud = np.concatenate((self.__frame_point_cloud, point_cloud), axis=1)



from __future__ import annotations

from src.datasets.dataset import Dataset
from src.datasets.frame_descriptor import FrameDescriptor


class WaymoSceneIterator(Dataset.SceneIterator):
    """Iterator over frames in a NuScenes scene.
    """

    def __init__(self,
                 scene_id: str,
                 scene_descriptor: dict):
        self.__scene_id = scene_id
        self.__scene_descriptor = scene_descriptor
        self.__frame_ids = sorted(scene_descriptor.keys())
        self.__current_frame = 0

    def __iter__(self) -> WaymoSceneIterator:
        """Reset iterator and returns itself.
        """
        self.__current_frame = 0
        return self

    def __next__(self) -> tuple[str, FrameDescriptor]:
        """Returns next frame.

        :return: tuple[str, dict[str, any]]
            Returns a tuple of frame id to frame meta-information
        """

        if self.__current_frame >= len(self.__frame_ids):
            raise StopIteration()

        frame_id = self.__frame_ids[self.__current_frame]
        frame_metadata = self.__scene_descriptor[frame_id]

        assert frame_id == frame_metadata['frame_id']

        instance_ids = frame_metadata['annos']['obj_ids']

        self.__current_frame += 1

        return frame_id, FrameDescriptor(frame_id=frame_id, instances_ids=instance_ids)


import os
import numpy as np

from pyquaternion import Quaternion

from src.utils.file_utils import list_all_files_with_extension
from src.utils.geometry_utils import points_in_box, transform_matrix


def find_all_scenes(dataset_root: str) -> list:
    """Returns list of scene ids.
    """
    metadata_files = list_all_files_with_extension(files=[dataset_root], extension='pkl')
    return [os.path.basename(file).split('.')[0] for file in metadata_files]


def load_scene_descriptor(dataset_root: str,
                          scene_id: str) -> dict:
    metadata_file = os.path.join(dataset_root, f"{scene_id}.pkl")

    assert os.path.exists(metadata_file), \
        f"Cannot find file {metadata_file}"

    raw_scene_descriptor = np.load(metadata_file, allow_pickle=True)
    return {frame_descriptor['frame_id']: frame_descriptor for frame_descriptor in raw_scene_descriptor}


def count_frames_in_scene(dataset_root: str,
                          scene_id: str) -> int:
    frame_pcr_dir = os.path.join(dataset_root, scene_id)
    pcr_list = list_all_files_with_extension(files=[frame_pcr_dir], extension='.npy', shallow=True)
    return len(pcr_list)


def get_frame_point_cloud(dataset_root: str,
                          scene_id: str,
                          frame_descriptor: dict) -> np.ndarray:
    frame_index = get_frame_index(frame_descriptor)
    frame_pcr_file = os.path.join(dataset_root, scene_id, f"{frame_index:04d}.npy")

    assert os.path.exists(frame_pcr_file), \
        f"Cannot find file {frame_pcr_file}"

    return np.load(frame_pcr_file).T


def get_instance_point_cloud(frame_point_cloud: np.ndarray,
                             instance_id: str,
                             frame_descriptor: dict) -> np.ndarray:
    """Returns point cloud for the given instance in the given frame.

    The returned point cloud has reset rotation and translation.

    :param frame_point_cloud: np.ndarray
        Frame point cloud in <dimension, N> format.
    :param instance_id: str
        ID of an instance.
    :param frame_descriptor: dict
        Descriptor of the given frame.
    :return: np.ndarray[float]
        Returns point cloud for the given object.
        Dimension of the array is 5xm.
    """
    annotations = frame_descriptor['annos']
    ids = annotations['obj_ids']

    # O(obj_ids)
    instance_index = np.where(ids == instance_id)
    instance_column = instance_index[0][0]

    center_xyz = annotations['location'][instance_column, :]
    dimensions_lwh = annotations['dimensions'][instance_column, :]
    heading_angle = annotations['heading_angles'][instance_column]

    points = frame_point_cloud[0:3, :]

    mask = points_in_box(center_xyz=center_xyz,
                         dimensions_lwh=dimensions_lwh,
                         heading_angle=heading_angle,
                         points=points)

    instance_point_cloud = frame_point_cloud[:, np.where(mask)[0]]

    identity_transformation = transform_matrix(center_xyz,
                                               Quaternion(angle=heading_angle, axis=[0, 0, 1]),
                                               inverse=True)

    instance_point_cloud = __apply_transformation_matrix(point_cloud=instance_point_cloud,
                                                         transformation_matrix=identity_transformation)
    return instance_point_cloud


def reapply_frame_transformation(point_cloud: np.ndarray,
                                 instance_id: str,
                                 frame_descriptor: dict) -> np.ndarray:
    annotations = frame_descriptor['annos']
    ids = annotations['obj_ids']

    instance_index = np.where(ids == instance_id)
    instance_column = instance_index[0][0]

    center_xyz = annotations['location'][instance_column, :]
    heading_angle = annotations['heading_angles'][instance_column]

    reverse_transformation = transform_matrix(center_xyz,
                                              Quaternion(angle=heading_angle, axis=[0, 0, 1]),
                                              inverse=False)

    instance_point_cloud = __apply_transformation_matrix(point_cloud=point_cloud,
                                                         transformation_matrix=reverse_transformation)

    return instance_point_cloud


def get_frame_index(frame_descriptor: dict) -> int:
    # Some of converted waymo formats contain frame_index,
    # while others use sample_idx inside of point_cloud
    # to figure out the frame index.
    point_cloud_descriptor = frame_descriptor['point_cloud']
    if 'frame_index' in point_cloud_descriptor:
        frame_index = point_cloud_descriptor['frame_index']
    elif 'sample_idx' in point_cloud_descriptor:
        frame_index = point_cloud_descriptor['sample_idx']
    else:
        raise Exception(f"Frame descriptor does not have frame_index. Descriptor: {frame_descriptor}")

    return frame_index


def __apply_transformation_matrix(point_cloud: np.ndarray,
                                  transformation_matrix: np.ndarray) -> np.ndarray:
    """Applies given transformation matrix to the given point cloud.

    :param point_cloud: np.ndarray[float]
        Point cloud to transform of shape [3, n]
    :param transformation_matrix: np.ndarray[float]
        Transformation matrix that describes rotation and translation of shape [4, 4].
    :return: np.ndarray[float]
        Modified point cloud.
    """
    points_count = point_cloud.shape[1]
    point_cloud[:3, :] = transformation_matrix.dot(
        np.vstack((point_cloud[:3, :], np.ones(points_count))))[:3, :]
    return point_cloud
