import torch
import kaolin.render.spc as spc_render
from wisp.utils import PerfTimer
from wisp.tracers import BaseTracer
from wisp.core import RenderBuffer


class PackedSPCTracer(BaseTracer):
    """Tracer class for sparse point clouds (packed rays).
    The logic of this tracer is straightforward and does not involve any neural operations:
    rays are intersected against the SPC points (cell centers).
    Each ray returns the color of the intersected cell, if such exists.
    """

    def __init__(self, **kwargs):
        """Set the default trace() arguments. """
        super().__init__(**kwargs)

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.

        Returns:
            (set): Set of channel strings.
        """
        return {"depth", "hit", "rgb", "alpha"}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.

        Returns:
            (set): Set of channel strings.
        """
        return {"rgb"}

    def trace(self, nef, channels, extra_channels, rays, lod_idx=None):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that
                            were not requested since those channels often had to be computed anyways.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        timer = PerfTimer(activate=False, show_memory=False)
        N = rays.origins.shape[0]

        # By default, SPCRFTracer will use the highest level of detail for the ray sampling.
        if lod_idx is None:
            lod_idx = nef.grid.blas.max_level

        ridx, pidx, depths = nef.grid.blas.raytrace(rays, lod_idx, with_exit=False)

        timer.check("Raytrace")

        # Get the indices of the ray tensor which correspond to hits
        first_hits_mask = spc_render.mark_pack_boundaries(ridx)
        first_hits_point = pidx[first_hits_mask]
        first_hits_ray = ridx[first_hits_mask]
        first_hits_depth = depths[first_hits_mask]

        # Get the color for each ray
        color = nef(ridx_hit=first_hits_point.long(), channels="rgb")

        timer.check("RGBA")
        del ridx, pidx, rays

        # Fetch colors and depth for closest hits
        ray_colors = color.squeeze(1)
        ray_depth = first_hits_depth
        depth = torch.zeros(N, 1, device=ray_depth.device)
        depth[first_hits_ray.long(), :] = ray_depth
        alpha = torch.ones([color.shape[0], 1], device=color.device)
        hit = torch.zeros(N, device=color.device).bool()

        # Populate the background
        rgb = torch.zeros(N, 3, device=color.device)
        out_alpha = torch.zeros(N, 1, device=color.device)
        color = alpha * ray_colors

        hit[first_hits_ray.long()] = alpha[..., 0] > 0.0
        rgb[first_hits_ray.long(), :3] = color
        out_alpha[first_hits_ray.long()] = alpha

        timer.check("Composit")

        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha)
