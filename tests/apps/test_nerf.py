# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from tests.test_utils import TestWispApp, run_wisp_script, collect_metrics_from_log, report_metrics


class TestNerfApp(TestWispApp):

    def test_hashgrid_lego_quick(self, lego_path, dataset_num_workers):
        cmd = 'app/nerf/main_nerf.py'
        cli_args = \
            f'--dataset-path {lego_path} ' \
            '--config app/nerf/configs/nerf_hash.yaml ' \
            f'--dataset-num-workers {dataset_num_workers} ' \
            '--mip 0 ' \
            '--num-steps 512 ' \
            '--raymarch-type ray ' \
            '--optimizer-type adam ' \
            '--hidden-dim 64 ' \
            '--epochs 200 ' \
            '--valid-every 100 ' \
            '--save-every -1 ' \
            '--render-tb-every -1 '

        out = run_wisp_script(cmd, cli_args)
        metrics = collect_metrics_from_log(out, ['PSNR'])

        assert float(metrics[100]['PSNR']) > 28.3, 'PSNR is too low.'
        assert float(metrics[200]['PSNR']) > 28.9, 'PSNR is too low.'
        # assert float(metrics[300]['PSNR']) > 29.3, 'PSNR is too low.'
        report_metrics(metrics)  # Prints to log


    def test_hashgrid_lego_best(self, lego_path, dataset_num_workers):
        cmd = 'app/nerf/main_nerf.py'
        cli_args = \
            f'--dataset-path {lego_path} ' \
            '--config app/nerf/configs/nerf_hash.yaml ' \
            f'--dataset-num-workers {dataset_num_workers} ' \
            '--mip 0 ' \
            '--num-steps 2048 ' \
            '--raymarch-type ray ' \
            '--optimizer-type rmsprop ' \
            '--hidden-dim 128 ' \
            '--epochs 100 ' \
            '--valid-every 100 ' \
            '--save-every -1 ' \
            '--render-tb-every -1 '

        out = run_wisp_script(cmd, cli_args)
        metrics = collect_metrics_from_log(out, ['PSNR'])

        assert float(metrics[100]['PSNR']) > 29.95, 'PSNR is too low.'
        # assert float(metrics[200]['PSNR']) > 31.5, 'PSNR is too low.'
        # assert float(metrics[300]['PSNR']) > 32.5, 'PSNR is too low.'
        report_metrics(metrics)  # Prints to log

    def test_hashgrid_V8(self, V8_path, dataset_num_workers):
        cmd = 'app/nerf/main_nerf.py'
        cli_args = \
            f'--dataset-path {V8_path} ' \
            '--config app/nerf/configs/nerf_hash.yaml ' \
            f'--dataset-num-workers {dataset_num_workers} ' \
            '--multiview-dataset-format rtmv ' \
            '--mip 2 ' \
            '--num-steps 16 ' \
            '--raymarch-type voxel ' \
            '--optimizer-type adam ' \
            '--hidden-dim 64 ' \
            '--epochs 10 ' \
            '--valid-every 10 ' \
            '--save-every -1 ' \
            '--render-tb-every -1 '

        out = run_wisp_script(cmd, cli_args)
        metrics = collect_metrics_from_log(out, ['PSNR'])

        assert float(metrics[10]['PSNR']) > 20.0, 'PSNR is too low.'
        report_metrics(metrics)  # Prints to log

    def test_octree_lego(self, lego_path, dataset_num_workers):
        cmd = 'app/nerf/main_nerf.py'
        cli_args = \
            f'--dataset-path {lego_path} ' \
            '--config app/nerf/configs/nerf_octree.yaml ' \
            f'--dataset-num-workers {dataset_num_workers} ' \
            '--mip 0 ' \
            '--num-steps 512 ' \
            '--raymarch-type ray ' \
            '--hidden-dim 64 ' \
            '--epochs 100 ' \
            '--valid-every 100 ' \
            '--save-every -1 ' \
            '--render-tb-every -1 '

        out = run_wisp_script(cmd, cli_args)
        metrics = collect_metrics_from_log(out, ['PSNR'])

        assert float(metrics[100]['PSNR']) > 28.4, 'PSNR is too low.'
        report_metrics(metrics)  # Prints to log

    def test_octree_V8(self, V8_path, dataset_num_workers):
        cmd = 'app/nerf/main_nerf.py'
        cli_args = \
            f'--dataset-path {V8_path} ' \
            '--config app/nerf/configs/nerf_octree.yaml ' \
            f'--dataset-num-workers {dataset_num_workers} ' \
            '--multiview-dataset-format rtmv ' \
            '--mip 2 ' \
            '--num-steps 16 ' \
            '--raymarch-type voxel ' \
            '--hidden-dim 128 ' \
            '--epochs 100 ' \
            '--valid-every 100 ' \
            '--save-every -1 ' \
            '--render-tb-every -1 '

        out = run_wisp_script(cmd, cli_args)
        metrics = collect_metrics_from_log(out, ['PSNR'])

        assert float(metrics[100]['PSNR']) > 28.1, 'PSNR is too low.'
        report_metrics(metrics)  # Prints to log

    def test_triplanar_lego(self, lego_path, dataset_num_workers):
        cmd = 'app/nerf/main_nerf.py'
        cli_args = \
            f'--dataset-path {lego_path} ' \
            '--config app/nerf/configs/nerf_triplanar.yaml ' \
            f'--dataset-num-workers {dataset_num_workers} ' \
            '--mip 2 ' \
            '--num-steps 512 ' \
            '--raymarch-type voxel ' \
            '--hidden-dim 128 ' \
            '--epochs 100 ' \
            '--valid-every 100 ' \
            '--save-every -1 ' \
            '--render-tb-every -1 '

        out = run_wisp_script(cmd, cli_args)
        metrics = collect_metrics_from_log(out, ['PSNR'])

        assert float(metrics[100]['PSNR']) > 30.5, 'PSNR is too low.'
        report_metrics(metrics)  # Prints to log

    def test_codebook_V8(self, V8_path, dataset_num_workers):
        cmd = 'app/nerf/main_nerf.py'
        cli_args = \
            f'--dataset-path {V8_path} ' \
            '--config app/nerf/configs/nerf_codebook.yaml ' \
            f'--dataset-num-workers {dataset_num_workers} ' \
            '--multiview-dataset-format rtmv ' \
            '--mip 2 ' \
            '--num-steps 16 ' \
            '--raymarch-type voxel ' \
            '--hidden-dim 128 ' \
            '--epochs 100 ' \
            '--valid-every 100 ' \
            '--save-every -1 ' \
            '--render-tb-every -1 '

        out = run_wisp_script(cmd, cli_args)
        metrics = collect_metrics_from_log(out, ['PSNR'])

        assert float(metrics[100]['PSNR']) > 27.5, 'PSNR is too low.'
        report_metrics(metrics)  # Prints to log
