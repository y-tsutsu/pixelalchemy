import threading

import gi
import numpy as np
from gi.repository import Gst

gi.require_version('Gst', '1.0')


class GStreamerPipeline:
    def __init__(self, path):
        Gst.init(None)

        pipeline_str = f'''
            filesrc location={path} !
            decodebin !
            videoconvert !
            video/x-raw,format=RGBA !
            appsink name=sink emit-signals=true sync=true max-buffers=1 drop=true
        '''

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name('sink')
        self.appsink.connect('new-sample', self.on_new_sample)

        self.frame = None
        self.width = None
        self.height = None

        self.lock = threading.Lock()

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        buf = sample.get_buffer()
        caps = sample.get_caps()

        structure = caps.get_structure(0)
        self.width = structure.get_value('width')
        self.height = structure.get_value('height')

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        data = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = data.reshape((self.height, self.width, 4))

        with self.lock:
            self.frame = frame.copy()

        buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def get_frame(self):
        with self.lock:
            return self.frame
