# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from tests.test_utils import TestWispApp, run_wisp_script, collect_metrics_from_log, report_metrics


class TestLatentNerfApp(TestWispApp):

    def test_latent_nerf_runs(self, lego_path, dataset_num_workers):
        cmd = 'examples/latent_nerf/main_demo.py'
        cli_args = f'--dataset-path {lego_path} --dataset-num-workers {dataset_num_workers} --epochs 1'

        out = run_wisp_script(cmd, cli_args)
        metrics = collect_metrics_from_log(out, ['PSNR'])
        report_metrics(metrics)  # Prints to log
