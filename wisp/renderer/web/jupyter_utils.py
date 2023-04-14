from __future__ import annotations
import contextlib
import glob
import io
import logging
import math
import os
import queue
import PIL.Image
import re
import threading
import time
import torch
import torchvision
from typing import Literal

try:
    from ipycanvas import hold_canvas
    from ipywidgets import Image as ImageWidget
except Exception:
    __IPYCANVAS_ERR = f'Wisp jupyter utilities require for you to install ipycanvas: pip install ipycanvas'
    logging.error(__IPYCANVAS_ERR)
    def hold_canvas(*args, **kwargs):
        raise RuntimeError(__IPYCANVAS_ERR)

from wisp.framework import WispState
from wisp.renderer.core import RendererCore
from wisp.renderer.core.control.io import WispMouseButton
from wisp.renderer.core.control.turntable import TurntableCameraMode

# Note: notebook does not have access to this, so we create a variable
WISP_ROOT_DIR = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, os.path.pardir))


@contextlib.contextmanager
def dummy_ctx_manager():
    yield None


# TODO(operel): this needs reworking of the renderer conventions to *not* resize the render buffer by default,
# because we want to send a small render buffer over the network.
# TODO(operel): It would also be great if this closure could take a camera object instead of keeping it in render_core;
# this way we can lock the camera, copy it, and render at ease instead of putting the lock around the whole
# render call. Stateless render layer would be able to do this.
# TODO(operel): Need easier way to specify custom camera to use with RendererCore. E.g., it never makes sense to
# have camera's width, height to be larger than the drawing canvas dimensions;
# OR -- make it easy to have camera.set_image_size
def make_render_closure(render_core: RendererCore, downscale_factor: int = 4):
    """Makes a render closure over input args, so render can be called without arguments.

    Args:
        render_core: the RendererCore to use for rendering
        downscale_factor: how much to downscale the image when rendering

    Returns: function() -> torch.Tensor 0..1 float, 4 x H x W, where H, W are determined by render_core.camera and
                the downscale_factor
    """
    def _render(td):
        """ Returns 0..1 float torch tensor of size 4 x H x W """
        rescale = None

        if downscale_factor == 1:
            render_core.set_full_resolution()
        else:
            render_core.set_low_resolution(downscale_factor)
            # TODO(operel): this should not be necessary
            rescale = torchvision.transforms.Resize((render_core.res_y, render_core.res_x))

        renderbuffer = render_core.render(time_delta=td)  # we only request rerender when dirty
        res = renderbuffer.image().rgba.permute(2, 0, 1) / 255  # C x H x W 0..1
        if rescale is not None:
            res = rescale(res)
        return res

    return _render


def save_canvas_render(canvas, filename, save_dir=None):
    """
    Convenience function to save a rendered frame to a default location, while appending a counter
    to the basename.

    Args:
        canvas: IpyCanvas canvas object
        filename: filename and extension where to save, note number will be appended `frame.png` --> `frame1.png`
        save_dir: directory where to save file; will use default _results/jupyter_renders if not provided
    """
    if save_dir is None:
        save_dir = os.path.join(WISP_ROOT_DIR, '_results', 'jupyter_renders')
    os.makedirs(save_dir, exist_ok=True)

    basename, extension = os.path.splitext(os.path.basename(filename))
    pattern = re.compile(r"%s(\d+)%s" % (basename, extension))
    fnames = [
        os.path.basename(x) for x in glob.glob(os.path.join(save_dir, "%s[0-9]*%s" % (basename, extension)))
    ]
    counts = [int(m.groups()[0]) for m in [pattern.match(f) for f in fnames] if m is not None]
    count = 0
    if len(counts) > 0:
        count = max(counts) + 1
    filename = os.path.join(save_dir, "%s%02d%s" % (basename, count, extension))
    logging.info(f'Saving frame to: {filename}')
    canvas.to_file(filename)


def np_img_to_compressed_bytes(np_img, format):
    """ Converts numpy array to bytes in the specified image format.

    Args:
        np_img: numpy array H x W x C uint8
        format: any format supported by Pillow, e.g. 'png' or 'jpeg' (note jpeg does not accept RGBA)
    Return:
        bytes
    """
    img = PIL.Image.fromarray(np_img)
    buff = io.BytesIO()
    img.save(buff, format=format)
    return buff.getvalue()


class LiveCanvasBase(object):
    """
    A wrapper for a set of events for the ipycanvas widget, primarily intended for mouse events.
    Will automatically start watching mouse_move events after mouse_down event.

    Subclasses should implement all or any of:
    _on_mouse_down(self, x, y)
    _on_mouse_up(self, x, y)
    _on_mouse_move(self, x, y)
    _on_key_down(self, key, shift, ctrl, meta)
    """

    def __init__(self, event_canvas):
        """
        Args:
            event_canvas: ipycanvas canvas object to which events will be bound
        """
        self.out = dummy_ctx_manager()  # Output widget
        self.event_canvas = event_canvas
        self.enable_output_on_move = True

    def output_to(self, out):
        """
        Set an output widget where logs and print statements will be written. Otherwise, all event
        handlers will be silent, even on error or exception.

        Args:
            out: ipywidgets output widget
        """
        if out is not None:
            self.out = out
        else:
            self.out = dummy_ctx_manager()

    def bind_events(self):
        """
        Binds canvas mouse events to the canvas provided to the constructor.
        """
        self.event_canvas.on_mouse_down(self.on_mouse_down)
        self.event_canvas.on_mouse_up(self.on_mouse_up)
        self.event_canvas.on_key_down(self.on_key_down)

    def unbind_events(self):
        """
        Unbinds all events.
        """
        self.event_canvas.on_mouse_down(self.on_mouse_down, remove=True)
        self.event_canvas.on_mouse_move(self.on_mouse_move, remove=True)
        self.event_canvas.on_mouse_up(self.on_mouse_up, remove=True)
        self.event_canvas.on_key_down(self.on_key_down, remove=True)

    def on_mouse_down(self, x, y):
        with self.out:
            self._on_mouse_down(x, y)
            # TODO: How to guarantee this is always removed?
            self.event_canvas.on_mouse_move(self.on_mouse_move)

    def on_mouse_up(self, x, y):
        with self.out:
            self._on_mouse_up(x, y)
            self.event_canvas.on_mouse_move(self.on_mouse_move, remove=True)

    def on_mouse_move(self, x, y):
        with self.out if self.enable_output_on_move else dummy_ctx_manager():
            self._on_mouse_move(x, y)

    def on_key_down(self, key, shift, ctrl, meta):
        with self.out:
            self._on_key_down(key, shift, ctrl, meta)

    def _on_mouse_down(self, x, y):
        pass

    def _on_mouse_up(self, x, y):
        pass

    def _on_mouse_move(self, x, y):
        pass

    def _on_key_down(self, key, shift, ctrl, meta):
        pass


class RenderDrawWithCameraThread(threading.Thread):
    """
    Separate thread to handle rendering and camera event queues by updating camera, doing the rendering and
    updating ipycanvas. This avoids, for example, calling render too frequently. Instead, we just add items
    to the queue if it is empty, and the renderer picks up render events at leisure.

    Used internally by LiveCameraControl; should not be instantiated separately.

    The logic of this class is complicated as the result of the stateful nature of the rendering closure.
    camera_queue - maintains all the camera mouse events
    render_queue - maintains at most 3 render events
    during each iteration:
        - process all accumulated camera events
        - process one render event
    """
    def __init__(self, cam_controller, event_canvas, render_closure, draw_canvas, highres_render_closure):
        super().__init__()
        self.controller = cam_controller
        self.xfactor = cam_controller.camera.width / event_canvas.width
        self.yfactor = cam_controller.camera.height / event_canvas.height

        self.position = None
        self.render_queue = queue.Queue(3)
        self.cam_queue = queue.Queue()
        self.render_closure = render_closure
        self.draw_canvas = draw_canvas
        self.highres_render_closure = highres_render_closure if highres_render_closure is not None else render_closure
        self.prev_render_time = None

    def _on_mouse_down(self, x, y):
        self.position = (float(x * self.xfactor), float(y * self.yfactor))
        self.controller.handle_mouse_press(self.position[0], self.position[1], WispMouseButton.LEFT_BUTTON)

    def _on_mouse_up(self, x, y):
        self.position = (float(x * self.xfactor), float(y * self.yfactor))
        self.controller.handle_mouse_release(self.position[0], self.position[1], WispMouseButton.LEFT_BUTTON)
        self.render_queue.put("highres", block=True, timeout=0.5)

    def _on_mouse_move(self, x, y):
        new_position = (float(x * self.xfactor), float(y * self.yfactor))
        dx = new_position[0] - self.position[0]
        dy = new_position[1] - self.position[1]
        self.position = new_position
        self.controller.handle_mouse_drag(self.position[0], self.position[1], dx, dy, WispMouseButton.LEFT_BUTTON)
        self.request_render_update()

    def _handle_zoom(self, amount):
        self.controller.zoom(amount)
        self.render_queue.put("highres", block=True, timeout=0.5)

    def process_camera_events(self):
        """Processes all accumulated camera events (can be quite a few if rendering is slow).
        Does not process new events that arrive after this is called."""
        for i in range(self.cam_queue.qsize()):
            task = self.cam_queue.get()
            task_type = task[0]
            if task_type == "zoom":
                self._handle_zoom(task[1])
            else:
                x = task[1]
                y = task[2]
                if task_type == "mouse_down":
                    self._on_mouse_down(x, y)
                elif task_type == "mouse_up":
                    self._on_mouse_up(x, y)
                elif task_type == "mouse_move":
                    self._on_mouse_move(x, y)
                else:
                    raise RuntimeError(f'Unsupported event type {task_type}')
            self.cam_queue.task_done()

    def process_render_event(self):
        """Processes a single render event (there can be at most 2 in the queue)."""
        try:
            task = self.render_queue.get(block=True, timeout=0.5)

            if task == "highres":
                self.prev_render_time = None
                res = self.highres_render_closure(None)
            else:
                start = time.time()
                res = self.render_closure(self.prev_render_time)
                end = time.time()
                self.prev_render_time = end - start

            # We expect res to be float32 0..1 torch tensor of size 4 x H x W; --> convert to numpy
            res = (res.permute(1, 2, 0) * 255).clip(min=0, max=255).to(torch.uint8).detach().cpu().numpy()

            # Update canvas
            with hold_canvas(self.draw_canvas):
                self.draw_canvas.clear_rect(0, 0, self.draw_canvas.width, self.draw_canvas.height)
                image = ImageWidget(value=np_img_to_compressed_bytes(res, 'png'))
                self.draw_canvas.draw_image(image, 0, 0, self.draw_canvas.width, self.draw_canvas.height)

            self.render_queue.task_done()

        except queue.Empty as e:
            pass

    def request_cam_update(self, event_type: Literal['mouse_down', 'mouse_up', 'mouse_move'], x, y):
        self.cam_queue.put((event_type, x, y))

    def request_zoom_update(self, amount):
        self.cam_queue.put(('zoom', amount, None))

    def request_render_update(self):
        """
        Requests the rendering thread to update the render.
        """
        try:
            if self.render_queue.qsize() < 2:
                self.render_queue.put("render", block=False)
        except queue.Full as e:
            # The queue already has a requested render
            pass

    def run(self):
        while True:
            self.process_camera_events()
            self.process_render_event()


class LiveCameraControl(LiveCanvasBase):
    """
    Connects jupyter notebook ipycanvas (HTML5 Canvas) events to a Wisp camera controller and
    a render closure.

    Minimal Jupyter example:

    from ipycanvas import Canvas
    from ipywidgets import VBox, Output
    import wisp.web.jupyter_utils

    # instantiate wisp_state, render_core, then ...

    out = Out()
    canvas = Canvas(width=500, height=500)
    turntable = wisp.web.jupyter_utils.LiveCameraControl.create_easy_turntable(wisp_state, render_core, canvas)
    turntable.activate(out)
    VBox((out, canvas))
    """
    def __init__(self, cam_controller, event_canvas, render_closure, draw_canvas=None, highres_render_closure=None):
        """
        Args:
            cam_controller (CameraControlMode): camera controller to hook up canvas events to
            event_canvas: ipycanvas object to bind events to
            render_closure: function ()-> (4 x H x W torch32 tensor 0..1) rendering closure to use by default,
                            any H x W is acceptable, regardless of drawing canvas size
            draw_canvas: ipycanvas to draw updated renders on, if different from event_canvas
            highres_render_closure: same as above, but only called on mouse_up
        """
        super().__init__(event_canvas)

        # Handle offscreen rendering
        self.render_closure = render_closure
        self.highres_render_closure = highres_render_closure if highres_render_closure is not None else render_closure
        self.event_thread = RenderDrawWithCameraThread(
            cam_controller, event_canvas, render_closure,
            draw_canvas=draw_canvas if draw_canvas is not None else event_canvas,
            highres_render_closure=self.highres_render_closure)
        self.event_thread.start()

    def test_rendering(self, highres=True):
        res = self.highres_render_closure(None) if highres else self.render_closure(None)
        res = (res.permute(1, 2, 0) * 255).clip(min=0, max=255).to(torch.uint8).detach().cpu().numpy()
        return res

    @classmethod
    def create_easy_turntable(cls, wisp_state: WispState, render_core: RendererCore,
                              event_canvas, draw_canvas=None) -> LiveCameraControl:
        """
        Convenience constructor for LiveCameraControl that creates a turntable mouse controller.

        Args:
            wisp_state: wisp state object
            render_core: render core object
            event_canvas: ipycanvas widget that will receive events (e.g. should be top canvas if using MultiCanvas)
            draw_canvas: ipycanvas widget where drawing should occur (defaults to event_canvas)
        Return:
            (LiveCameraControl) with events not yet bound (call `activate`)
        """
        wisp_state.renderer.reference_grids = ['xz']
        WispMouseButton.register_symbol(WispMouseButton.LEFT_BUTTON, 1)
        controller = TurntableCameraMode(render_core, wisp_state)

        render_closure = make_render_closure(render_core, downscale_factor=4)
        highres_render_closure = make_render_closure(render_core, downscale_factor=1)
        jupyter_turntable = LiveCameraControl(
            controller, event_canvas, render_closure, draw_canvas=draw_canvas,
            highres_render_closure=highres_render_closure)
        return jupyter_turntable

    def activate(self, out=None, enable_output_on_move=True, render_now=True):
        """
        Runs default event binding (note: in some cases it's desirable to handle events differently, so this
        is not part of the constructor), sets up output widget.

        Args:
            out: ipywidgets Out output widget or None -- will capture all logs/pints from callbacks
            enable_output_on_move: if False, will not capture outputs in mouse move callbacks -- this
               is faster but can make debugging difficult, so we enable it by default.
            render_now: if to update the canvas with the render
        """
        self.output_to(out)
        self.enable_output_on_move = enable_output_on_move
        self.bind_events()
        if render_now:
            self.request_render_update()

    def request_render_update(self):
        try:
            self.event_thread.render_queue.put("highres", block=False)
        except queue.Full as e:
            pass  # update already requested

    def request_cam_update(self, event_type: Literal['mouse_down', 'mouse_up', 'mouse_move'], x, y):
        self.event_thread.cam_queue.put((event_type, x, y))

    def zoom(self, amount):
        self.event_thread.request_zoom_update(amount)

    def _on_mouse_down(self, x, y):
        self.event_thread.request_cam_update('mouse_down', x, y)

    def _on_mouse_up(self, x, y):
        self.event_thread.request_cam_update('mouse_up', x, y)

    def _on_mouse_move(self, x, y):
        self.event_thread.request_cam_update('mouse_move', x, y)

    def _on_key_down(self, key, shift, ctrl, meta):
        pass
