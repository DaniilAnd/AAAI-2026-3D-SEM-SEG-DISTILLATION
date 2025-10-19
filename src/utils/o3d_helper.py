import open3d as o3d
import numpy as np


def convert_to_o3d_pointcloud(
        points: np.ndarray) -> o3d.geometry.PointCloud:
    """Converts a point cloud in numpy array format to an Open3D point cloud.

    :param points: np.ndarray
        Point cloud as a mxk float numpy array with columns X, Y, Z, intensity.
    :return: o3d.geometry.PointCloud
        Open3D point cloud object with intensity-based coloring.
    """

    # xyz = points[:3, :].T
    # intensity = points[3, :].reshape(-1, 1)

    xyz = points[:, :3]
    colors = points[:, 3:6]
    
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(xyz)

    
    o3d_pc.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pc

# TODO: add o3d downsampling (same base)
def downsampling_pc(
        o3d_pc: o3d.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
    """Downsampling an Open3D point cloud by o3d_pc.voxel_down_sample.

    :param o3d_pc: o3d.geometry.PointCloud
        Open3D point cloud object.
    :voxel_size o3d_pc: float
        voxel_size
    :return: o3d.geometry.PointCloud
        Downsampled point cloud
    """
    return o3d_pc.voxel_down_sample(voxel_size=voxel_size)


def convert_to_numpy_array(
        o3d_pc: o3d.geometry.PointCloud) -> np.ndarray:
    """Converts an Open3D point cloud to a numpy array.

    :param o3d_pc: o3d.geometry.PointCloud
        Open3D point cloud object.
    :return: np.ndarray[float]
        Point cloud as a mxk float numpy array with columns X, Y, Z, intensity.
    """
    xyz = np.asarray(o3d_pc.points).T
    colors = np.asarray(o3d_pc.colors).T
    point_cloud_np = np.vstack((xyz, colors))  # .T
    return point_cloud_np
