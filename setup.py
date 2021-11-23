from setuptools import find_packages, setup

import os
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='mmdet3d-gaussian',
        version='0.1.0',
        description=('mmdet3d-gaussian'),
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='zhouyue&zhanggefan',
        author_email='lizaozhouke@sjtu.edu.cn',
        keywords='computer vision, 3D object detection',
        packages=find_packages(),
        include_package_data=True,
        license='Apache License 2.0',
        ext_modules=[
            make_cuda_ext(
                name='voxel_utils',
                module='mmdet3d_gaussian.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cuda.cu'
                ]),
            make_cuda_ext(
                name='vsa_utils',
                module='mmdet3d_gaussian.ops.vsa',
                sources=[
                    'src/ball_query.cu',
                    'src/group_points.cu',
                    'src/sampling.cu',
                    'src/voxel_query.cpp',
                    'src/voxel_query_gpu.cu',
                    'src/vsa_utils.cpp',
                ]),
            make_cuda_ext(
                name='eval_utils',
                module='mmdet3d_gaussian.ops.eval',
                sources=[
                    'eval_utils.cpp',
                    'matcher.cpp',
                    'affinity.cpp'
                ])],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
