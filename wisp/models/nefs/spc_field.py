from typing import Dict, Any
import numpy as np
import torch
from wisp.models.grids import OctreeGrid
from wisp.models.nefs.nerf import NeuralRadianceField, BaseNeuralField
import wisp.ops.spc as wisp_spc_ops
import kaolin.ops.spc as kaolin_ops_spc


class SPCField(BaseNeuralField):
    """ A field based on Structured Point Clouds (SPC) from kaolin.
    SPC is a hierarchical compressed data structure, which can be interpreted in various ways:
    * Quantized point cloud, where each sparse point is quantized to some (possibly very dense) grid.
      Each point is associated with some feature(s).
    * An Octree, where each cell center is represented by a quantized point.
    Throughout wisp, SPCs are used to implement efficient octrees or grid structures.
    This field class allows wisp to render SPCs directly with their feature content (hence no embedders or decoders
    are assumed).

    When rendered, SPCs behave like octrees which allow for efficient tracing.
    Feature samples per ray may be collected from each intersected "cell" of the structured point cloud.
    """

    def __init__(self, spc_octree, features_dict=None, device='cuda'):
        r"""Creates a new Structured Point Cloud (SPC), represented as a Wisp Field.

        In wisp, SPCs are considered neural fields, since their features may be optimized.
        See `examples/spc_browser` for an elaborate description of SPCs.

        Args:
            spc_octree (torch.ByteTensor):
                A tensor which holds the topology of the SPC.
                Each byte represents a single octree cell's occupancy (that is, each bit of that byte represents
                the occupancy status of a child octree cell), yielding 8 bits for 8 cells.
                See also https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html
            features_dict (dict):
                A dictionary holding the features information of the SPC.
                Keys are assumed to be a subset of ('colors', 'normals').
                Values are torch feature tensors containing information per point, of shape
                :math:`(\text{num_points}, \text{feat_dim})`.
                Where `num_points` is the number of occupied cells in the SPC.
                See `kaolin.ops.conversions.pointcloud.unbatched_pointcloud_to_spc` for conversion of point
                cloud information to such features.
            device (torch.device):
                Torch device on which the features and topology of the SPC field will be stored.
        """
        super().__init__()
        self.spc_octree = spc_octree
        self.features_dict = features_dict if features_dict is not None else dict()
        self.spc_device = device
        self.grid = None
        self.colors = None
        self.normals = None
        self.init_grid(spc_octree)

    def init_grid(self, spc_octree):
        """ Uses the OctreeAS / OctreeGrid mechanism to quickly parse the SPC object into a Wisp Neural Field.

        Args:
            spc_octree (torch.ByteTensor):
                A tensor which holds the topology of the SPC.
                Each byte represents a single octree cell's occupancy (that is, each bit of that byte represents
                the occupancy status of a child octree cell), yielding 8 bits for 8 cells.
                See also https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html
        """
        # Use features, either colors or normals
        spc_features = self.features_dict
        if "colors" in self.features_dict:
            colors = spc_features["colors"]
            colors = colors.reshape(-1, 4) / 255.0
            self.colors = colors
        if "normals" in self.features_dict:
            normals = spc_features["normals"]
            normals = normals.reshape(-1, 3)
            self.normals = normals

        if self.colors is None:  # If no color features exist, use normals information
            if self.normals is not None:
                colors = 0.5 * (normals + 1.0)
            else:
                # manufacture colors from point coordinates
                lengths = torch.tensor([len(spc_octree)], dtype=torch.int32)
                level, pyramids, exsum = kaolin_ops_spc.scan_octrees(spc_octree, lengths)
                point_hierarchies = kaolin_ops_spc.generate_points(spc_octree, pyramids, exsum)
                # get coordinate of highest level
                colors = point_hierarchies[pyramids[0,1,level]:]
                # normalize
                colors = colors/np.power(2, level)

            self.colors = colors

        # By default assume the SPC keeps features only at the highest LOD.
        # Compute the highest LOD:
        _, pyramid, _ = wisp_spc_ops.octree_to_spc(spc_octree)
        max_level = pyramid.shape[-1] - 2

        self.grid = OctreeGrid.from_spc(
            spc_octree=spc_octree,
            feature_dim=3,
            base_lod=max_level,
            num_lods=0  # SPCFields track features internally, avoiding full OctreeGrid initialization is faster
        )

    @property
    def device(self):
        """ Returns the device used to process inputs in this neural field.

        Returns:
            (torch.device): The expected device used for this Structured Point Cloud.
        """
        return self.spc_device

    def register_forward_functions(self):
        """Register the forward functions.
        """
        # TODO (operel): support normals channel explicitly
        self._register_forward_function(self.rgba, ["rgb"])

    def rgba(self, ridx_hit=None):
        """Compute color for the provided ray hits.

        Args:
            ridx_hit (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     used to indicate index of first hit voxel.

        Returns:
            {"rgb": torch.FloatTensor}:
                - RGB tensor of shape [batch, 1, 3]
        """
        # find offset to final level to make indices relative to final level
        level = self.grid.blas.max_level
        offset = self.grid.blas.pyramid[1, level]
        ridx_hit = ridx_hit - offset

        colors = self.colors[ridx_hit, :3].unsqueeze(1)
        return dict(rgb=colors)

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        properties = {
            "Grid": self.grid,
        }
        return properties
