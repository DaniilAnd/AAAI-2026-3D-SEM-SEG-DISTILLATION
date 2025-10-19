from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional
import numpy as np
from typing import Optional, Iterable
from abc import ABC, abstractmethod
from pyquaternion import Quaternion
import os
import numpy as np
from typing import Optional, Iterable, List
from abc import ABC, abstractmethod
import shutil

def list_all_files_with_extension(files: list,
                                  extension: str,
                                  shallow: bool = False) -> list:
    result = list()

    for file in files:
        if os.path.isfile(file) and file.endswith(f'.{extension}'):
            result.append(file)
        elif os.path.isdir(file) and not shallow:
            sub_files = [os.path.join(file, f) for f in os.listdir(file)]
            result.extend(list_all_files_with_extension(files=sub_files,
                                                        extension=extension,
                                                        shallow=shallow))

    return result

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def __corners(centers_xyz: np.ndarray,
              sizes_lwh: np.ndarray,
              orientation: Quaternion) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    l, w, h = sizes_lwh

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(orientation.rotation_matrix, corners)

    # Translate
    x, y, z = centers_xyz
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def points_in_box(center_xyz: np.ndarray,
                  dimensions_lwh: np.ndarray,
                  heading_angle: float,
                  points: np.ndarray):
    """Specifies the points of the point cloud which are inside the bounding box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579.

    Runtime complexity is O(N*d).

    :param center_xyz: np.ndarray
        Coordinates of the bounding box center.
    :param dimensions_lwh: np.ndarray
        Length, width, and height of the bounding box.
    :param heading_angle: float
        Heading angle (i.e. z-rotation) of the bounding box.
    :param points: np.ndarray
        Frame point loud.
    :return: <np.bool: n, >.
    """

    orientation = Quaternion(angle=heading_angle, axis=[0, 0, 1])
    corners = __corners(center_xyz, dimensions_lwh, orientation)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask


def apply_transformation_matrix(point_cloud: np.ndarray,
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

class FramePatcher(ABC):
    """Patches a frame with new point cloud.
    """

    @property
    @abstractmethod
    def frame_id(self) -> str:
        """Returns processed frame id.

        :return: str
            ID of a frame, should be unique across a scene.
        """
        ...

    @property
    @abstractmethod
    def frame(self) -> np.ndarray:
        """Returns frame point cloud.

        :return: np.ndarray[float]
            Returns numpy array of kxm, where m is samples count.
        """
        ...

    @abstractmethod
    def patch_instance(self,
                       instance_id: str,
                       point_cloud: np.ndarray):
        """Replaces a point cloud of the instance in the frame with the given point cloud.

        :param instance_id: str
            Instance ID.
        :param point_cloud: np.ndarray[float]
            New point cloud.
        """
        ...

class FrameDescriptor(object):
    def __init__(self,
                 frame_id: str,
                 instances_ids: list):
        self.__frame_id = frame_id
        self.__instances_ids = instances_ids

    @property
    def frame_id(self) -> str:
        return self.__frame_id

    @property
    def instances_ids(self) -> list:
        return self.__instances_ids

class Dataset(ABC):
    class SceneIterator(ABC, Iterable):
        """Iterates through all frames in a scene.

        The complexity of __init__ method should not
        exceed O(1).
        """

        @abstractmethod
        def __next__(self):
            """Retrieves information about the next frame.

            Runtime complexity of getting the next frame information is O(1).

            :raises:
                StopIteration when there are no elements left.
            :return:
                A tuple of frame_id (string) and associated frame meta-information,
                grouped in FrameDescriptor class.
            """
            ...

    @property
    @abstractmethod
    def dataroot(self) -> str:
        ...

    @property
    @abstractmethod
    def scenes(self) -> list:
        ...

    @abstractmethod
    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        """Returns a scene iterator for the given scene_id.

        Scene iterator navigates through all available frames.

        Creating a scene iterator is a lightweight operation:
        performance of the method should not exceed O(1).

        :param scene_id: str
            Unique scene identifier.
        :return:
            An instance of SceneIterator.
        """
        ...

    @abstractmethod
    def load_frame_patcher(self, scene_id: str, frame_id: str) -> FramePatcher:
        ...

    @abstractmethod
    def can_serialise_frame_point_cloud(self,
                                        scene_id: str,
                                        frame_id: str) -> bool:
        """Checks whether it is possible to serialise a point cloud for the given frame in the scene.

        It is possible to serialise a point cloud for the frame if there is no
        serialised version of the same point cloud on disk.

        Runtime complexity is O(1).

        :param scene_id: str
            Unique scene identifier.
        :param frame_id: str
            Unique frame identifier.
        :return:
            True if it is possible to serialise the point cloud and
            False otherwise.
        """
        ...

    @abstractmethod
    def serialise_frame_point_clouds(self,
                                     scene_id: str,
                                     frame_id: str,
                                     frame_point_cloud: np.ndarray) -> Optional[str]:
        ...

    @abstractmethod
    def get_frame_point_cloud(self,
                              scene_id: str,
                              frame_id: str) -> np.ndarray:
        """Loads frame point cloud.

        Usually, loads the point cloud in memory.
        Potentially, extremely heavy operation.
        Runtime consideration is at least O(N*d),
        where N is the number of point in the point cloud
        and D is their dimensionality.

        It is reasonable to expect N >= 1000 and d >= 3.
        """
        ...

    @abstractmethod
    def get_instance_point_cloud(self,
                                 scene_id: str,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        """Returns point cloud of the specified instance.

        Runtime complexity is O(N*d).

        """
        ...


class PointceptFramePatcher(FramePatcher):
    def __init__(self, frame_dir: str, dataset: PointceptDataset):
        self.frame_dir = frame_dir
        self.dataset = dataset
        self._pointcloud = self._load_pointcloud()
        # Инициализируем сегменты: загружаем или создаем массив -1
        segment_path = os.path.join(self.frame_dir, "segment.npy")
        if os.path.exists(segment_path):
            self._segments = np.load(segment_path)
        else:
            num_points = self._pointcloud.shape[1]
            self._segments = np.full(num_points, -1, dtype=np.int32)

    def _load_pointcloud(self) -> np.ndarray:
        coord = np.load(os.path.join(self.frame_dir, "coord.npy"))  # (N, 3)
        strength = np.load(os.path.join(self.frame_dir, "strength.npy"))  # (N, 1)
        return np.vstack([coord.T, strength.T])  # (4, N)

    @property
    def frame_id(self) -> str:
        return os.path.basename(self.frame_dir)

    @property
    def frame(self) -> np.ndarray:
        return self._pointcloud


    def patch_instance(self, instance_id: str, point_cloud: np.ndarray):
        target_pose = np.load(os.path.join(self.frame_dir, "pose.npy"))
        local_points = self.dataset.reapply_frame_transformation(point_cloud, target_pose)
        
        # Создаем маску для текущего экземпляра
        mask = self._segments == int(instance_id)
        
        # Удаляем старые точки экземпляра
        self._pointcloud = np.hstack([self._pointcloud[:, ~mask], local_points])
        
        # Обновляем сегменты: удаляем старые, добавляем новые с instance_id
        new_segments = np.full(local_points.shape[1], int(instance_id), dtype=np.int32)
        self._segments = np.hstack([self._segments[~mask], new_segments])

    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Преобразует точки с использованием матрицы преобразования.
        
        Args:
            points: [3, N] массив точек
            transform: [4, 4] матрица преобразования
            
        Returns:
            [3, N] преобразованные точки
        """
        homogeneous = np.vstack([points, np.ones((1, points.shape[1]))])
        return (transform @ homogeneous)[:3, :]


class PointceptSceneIterator(Dataset.SceneIterator):
    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.frames = sorted([f for f in os.listdir(scene_path) 
                            if os.path.isdir(os.path.join(scene_path, f))])
        self.index = 0

    def __iter__(self) -> 'PointceptSceneIterator':
        self.index = 0
        return self

    def __next__(self) -> tuple[str, FrameDescriptor]:
        if self.index >= len(self.frames):
            raise StopIteration()
        
        frame_id = self.frames[self.index]
        frame_dir = os.path.join(self.scene_path, frame_id)
        instance_ids = self._load_instance_ids(frame_dir)
        descriptor = FrameDescriptor(frame_id, instance_ids)
        
        self.index += 1
        return frame_id, descriptor

    def _load_instance_ids(self, frame_dir: str) -> List[int]:
        segment_path = os.path.join(frame_dir, "segment.npy")
        excluded_classes = {-1, 0, 1, 2, 4, 5, 9, 10, 12, 15, 20, 21, 17, 13}  # excluded classes
        
        if os.path.exists(segment_path):
            segments = np.load(segment_path)
            unique_ids = np.unique(segments)
            # filtering
            return [int(id) for id in unique_ids if int(id) not in excluded_classes]
        return []


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


class PointceptDataset(Dataset):
    def __init__(self, dataset_root: str):
        self._dataset_root = dataset_root
        self._scenes = self._find_scenes()
        self._frame_patcher_cache = {}

    def _find_scenes(self) -> List[str]:
        """Находит все сцены в датасете."""
        scenes = []
        for split in ["training", "validation", "testing"]:
            split_dir = os.path.join(self._dataset_root, split)
            if os.path.exists(split_dir):
                scenes.extend([
                    d for d in os.listdir(split_dir)
                    if os.path.isdir(os.path.join(split_dir, d))
                ])
        return scenes

    @property
    def dataroot(self) -> str:
        return self._dataset_root

    @property
    def scenes(self) -> List[str]:
        return self._scenes
    
    def reapply_frame_transformation(self, point_cloud: np.ndarray, target_frame_pose: np.ndarray) -> np.ndarray:
        """Преобразует точки обратно в локальные координаты целевого кадра"""
        inv_pose = np.linalg.inv(target_frame_pose)
        transformed_points = self._transform_points(point_cloud[:3, :], inv_pose)
        return np.vstack([transformed_points, point_cloud[3:, :]])
    
    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        """Возвращает итератор по кадрам сцены."""
        scene_path = self._get_scene_path(scene_id)
        return PointceptSceneIterator(scene_path)
    
    def _find_original_path(self, scene_id: str, frame_id: str) -> tuple:
        for split in ["training", "validation", "testing"]:
            potential_path = os.path.join(self._dataset_root, split, scene_id, frame_id)
            if os.path.exists(potential_path):
                return potential_path, split
        return None, None
    
    def load_frame_patcher(self, scene_id: str, frame_id: str) -> FramePatcher:
        frame_dir = self._get_frame_dir(scene_id, frame_id)
        patcher = PointceptFramePatcher(frame_dir, self)
        self._frame_patcher_cache[(scene_id, frame_id)] = patcher
        return patcher

    def can_serialise_frame_point_cloud(self, scene_id: str, frame_id: str) -> bool:
        """Проверяет возможность сериализации облака точек."""
        frame_dir = self._get_frame_dir(scene_id, frame_id)
        patched_dir = os.path.join(self._dataset_root, 'patched_v3', 
                                 os.path.basename(os.path.dirname(os.path.dirname(frame_dir))),
                                 scene_id, frame_id)
        return not os.path.exists(os.path.join(patched_dir, "coord.npy"))

    def serialise_frame_point_clouds(self, scene_id: str, frame_id: str, 
                                   frame_point_cloud: np.ndarray) -> Optional[str]:
        frame_dir = self._get_frame_dir(scene_id, frame_id)
        split_name = os.path.basename(os.path.dirname(os.path.dirname(frame_dir)))
        
        patched_dir = os.path.join(self._dataset_root, 'patched_v3', split_name, 
                                  scene_id, frame_id)
        os.makedirs(patched_dir, exist_ok=True)

        np.save(os.path.join(patched_dir, "coord.npy"), frame_point_cloud[:3, :].T)
        np.save(os.path.join(patched_dir, "strength.npy"), frame_point_cloud[3:, :].T)

        # proces segment
        if (scene_id, frame_id) in self._frame_patcher_cache:
            segments = self._frame_patcher_cache[(scene_id, frame_id)]._segments
        else:
            original_segment = os.path.join(frame_dir, "segment.npy")
            segments = np.load(original_segment) if os.path.exists(original_segment) else \
                      np.full(frame_point_cloud.shape[1], -1, dtype=np.int32)
        np.save(os.path.join(patched_dir, "segment.npy"), segments)

        # copy seconds file
        for file in ["pose.npy"]:
            src = os.path.join(frame_dir, file)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(patched_dir, file))

        return patched_dir

    def get_frame_point_cloud(self, scene_id: str, frame_id: str) -> np.ndarray:
        """Загружает облако точек кадра."""
        frame_dir = self._get_frame_dir(scene_id, frame_id)
        coord = np.load(os.path.join(frame_dir, "coord.npy"))
        strength = np.load(os.path.join(frame_dir, "strength.npy"))
        return np.vstack([coord.T, strength.T])

    def get_instance_point_cloud_old(self, scene_id: str, frame_id: str, instance_id: str, frame_point_cloud: np.ndarray) -> np.ndarray:
        """Извлекает облако точек для конкретного экземпляра."""
        frame_dir = None
        for split in ["training", "validation", "testing"]:
            potential_path = os.path.join(self._dataset_root, split, scene_id, frame_id)
            if os.path.exists(potential_path):
                frame_dir = potential_path
                break
                
        if frame_dir is None:
            raise ValueError(f"Frame {frame_id} not found in scene {scene_id}")
            
        # load segment
        segment_path = os.path.join(frame_dir, "segment.npy")
        if not os.path.exists(segment_path):
            raise ValueError(f"Segment file not found for frame {frame_id} in scene {scene_id}")
            
        segments = np.load(segment_path)
        
        # create mask
        mask = segments == int(instance_id)
        
        return frame_point_cloud[:, mask]
    
    def get_instance_point_cloud(self, scene_id: str, frame_id: str, 
                               instance_id: str, frame_point_cloud: np.ndarray) -> np.ndarray:
        """Возвращает точки экземпляра в глобальных координатах."""
        # Получаем маску точек экземпляра
        frame_dir = self._get_frame_dir(scene_id, frame_id)
        segments = np.load(os.path.join(frame_dir, "segment.npy"))
        mask = segments == int(instance_id)
        instance_points = frame_point_cloud[:, mask]

        # Преобразуем в глобальные координаты
        pose = np.load(os.path.join(frame_dir, "pose.npy"))
        global_coords = self._transform_points(instance_points[:3, :], pose)
        
        return np.vstack([global_coords, instance_points[3:, :]])
    
    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Преобразует точки с использованием матрицы преобразования.
        
        Args:
            points: [3, N] массив точек
            transform: [4, 4] матрица преобразования
            
        Returns:
            [3, N] преобразованные точки
        """
        homogeneous = np.vstack([points, np.ones((1, points.shape[1]))])
        return (transform @ homogeneous)[:3, :]
    
    def _get_frame_dir(self, scene_id: str, frame_id: str) -> str:
        """Возвращает путь к кадру."""
        for split in ["training", "validation", "testing"]:
            path = os.path.join(self.dataroot, split, scene_id, frame_id)
            if os.path.exists(path):
                return path
        raise ValueError(f"Frame {frame_id} not found in scene {scene_id}")
    
    def _get_scene_path(self, scene_id: str) -> str:
        """Возвращает полный путь к сцене."""
        for split in ["training", "validation", "testing"]:
            path = os.path.join(self.dataroot, split, scene_id)
            if os.path.exists(path):
                return path
        raise ValueError(f"Scene {scene_id} not found in dataset")

    def align_points_to_frame(self, points: np.ndarray, 
                             source_pose: np.ndarray, 
                             target_frame_id: str, 
                             target_scene_id: str) -> np.ndarray:
        """Выравнивает точки относительно целевого кадра.
        
        Args:
            points: [3, N] точки в системе source_pose
            source_pose: [4, 4] исходная матрица преобразования
            target_frame_id: ID целевого кадра
            target_scene_id: ID сцены целевого кадра
            
        Returns:
            [3, N] точки в системе координат целевого кадра
        """
        target_pose = np.load(os.path.join(
            self._get_frame_dir(target_scene_id, target_frame_id), "pose.npy"))
        
        # Вычисляем относительное преобразование
        relative_transform = np.linalg.inv(target_pose) @ source_pose
        return self._transform_points(points, relative_transform)