import numpy as np
from typing import Iterator, Tuple


class DensityCalculator:
    def __init__(self, filename):
        self.gammaonly = False
        self.ngm_g = 0
        self.nspin = 0
        self.bmat = np.zeros((3, 3))
        self.miller_indices = None
        self.rhog = None
        self.cell_volume = None
        self.read_binary_file(filename)

    def read_binary_file(self, filename: str):
    # the func will be further modified when PR 4991 are merged.
        with open(filename, 'rb') as f:
            # Read header
            self.gammaonly = np.fromfile(f, dtype=bool, count=1)[0]
            self.ngm_g, self.nspin = np.fromfile(f, dtype=np.int32, count=2)
            
            # Read reciprocal lattice vectors
            self.bmat = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
            
            # Read Miller indices
            self.miller_indices = np.fromfile(f, dtype=np.int32, count=self.ngm_g*3).reshape(self.ngm_g, 3)
            
            # Read rhog
            self.rhog = np.fromfile(f, dtype=np.complex128, count=self.ngm_g)
            
            # If nspin == 2, read second spin component (we'll ignore it for now)
            if self.nspin == 2:
                _ = np.fromfile(f, dtype=np.complex128, count=self.ngm_g)

        # Calculate cell volume and G vectors
        self.cell_volume = np.abs(np.linalg.det(self.bmat))
        self.g_vectors = 2 * np.pi * np.dot(self.miller_indices, self.bmat.T)

    def calculate_density_batch(self, points: np.ndarray) -> np.ndarray:
        phases = np.exp(1j * np.dot(points, self.g_vectors.T))
        densities = np.real(np.dot(phases, self.rhog)) / self.cell_volume
        return densities


def generate_grid(lattice_vectors: np.ndarray, 
                  grid_size: Tuple[int, int, int], 
                  origin: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    生成给定晶格和网格大小的实空间网格点。

    参数:
    lattice_vectors (np.ndarray): 形状为 (3, 3) 的晶格向量
    grid_size (Tuple[int, int, int]): 三个方向上的网格点数
    origin (np.ndarray): 网格原点坐标，默认为 (0, 0, 0)

    返回:
    np.ndarray: 形状为 (N, 3) 的网格点坐标数组
    """
    nx, ny, nz = grid_size
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    z = np.linspace(0, 1, nz, endpoint=False)
    
    grid = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack(grid, axis=-1).reshape(-1, 3)
    
    # 将分数坐标转换为实空间坐标
    real_points = np.dot(points, lattice_vectors) + origin
    
    return real_points


def calculate_density(filename: str, 
                      grid_size: Tuple[int, int, int], 
                      origin: np.ndarray = np.zeros(3), 
                      batch_size: int = 1000) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    计算给定网格的电子密度，以迭代器形式返回结果。

    参数:
    filename (str): 包含 n(G) 数据的二进制文件路径
    grid_size (Tuple[int, int, int]): 三个方向上的网格点数
    origin (np.ndarray): 网格原点坐标，默认为 (0, 0, 0)
    batch_size (int): 每批处理的点数

    返回:
    Iterator[Tuple[np.ndarray, np.ndarray]]: yielding (坐标, 密度值) 对
    """
    calculator = DensityCalculator(filename)
    
    points = generate_grid(calculator.lattice_vectors, grid_size, origin)
    
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        batch_densities = calculator.calculate_density_batch(batch_points)
        yield batch_points, batch_densities
