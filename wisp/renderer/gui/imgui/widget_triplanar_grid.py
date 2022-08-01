import numpy as np
import imgui
from wisp.framework import WispState
from wisp.models.grids import TriplanarGrid
from .widget_imgui import WidgetImgui
from .widget_property_editor import WidgetPropertyEditor


class WidgetTriplanarGrid(WidgetImgui):
    def __init__(self):
        super().__init__()
        self.properties_widget = WidgetPropertyEditor()

    def paint(self, state: WispState, triplanar_grid: TriplanarGrid = None, *args, **kwargs):
        if triplanar_grid is not None:
            properties = {
                "Feature Dims": triplanar_grid.feature_dim,
                "Total LODs": triplanar_grid.max_lod,
                "Active feature LODs": ', '.join([str(x) for x in triplanar_grid.active_lods]),
                "Interpolation": triplanar_grid.interpolation_type,
                "Multiscale aggregation": triplanar_grid.multiscale_type
            }
            self.properties_widget.paint(state=state, properties=properties)

            # pyramid = triplanar_grid.blas.pyramid
            # if pyramid is not None and pyramid.shape[1] > 1:
            #     points_per_lod = pyramid[0, :-2].cpu().numpy()
            #     imgui.text(f"Occupancy per LOD (%):")
            #     occupancy_hist = [occupied_cells / 8**lod for lod, occupied_cells in enumerate(points_per_lod)]
            #     width, height = imgui.get_content_region_available()
            #     imgui.plot_histogram(label="##octree_grid", values=np.array(occupancy_hist, dtype=np.float32),
            #                          graph_size=(width, 20))