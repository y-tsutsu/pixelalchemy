import moderngl
import moderngl_window as mglw
import numpy as np


class GLRenderer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = 'GStreamer + OpenGL Edge detection'
    window_size = (800, 600)
    pipeline = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Quad
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
            +1.0, -1.0, 1.0, 0.0,
            -1.0, +1.0, 0.0, 1.0,
            +1.0, +1.0, 1.0, 1.0,
        ], dtype='f4')

        # Vertex Buffer Object
        self.vbo = self.ctx.buffer(vertices.tobytes())

        # 仮テクスチャ（あとでサイズ更新）
        self.texture = None

        with open('shaders/vertex.vert') as fvert, open('shaders/fragment.frag') as ffrag:
            self.prog = self.ctx.program(vertex_shader=fvert.read(), fragment_shader=ffrag.read())

        # Vertex Array Object
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f', 'in_pos', 'in_uv')]
        )

    def on_render(self, time, frame_time):
        self.ctx.clear(0.0, 0.0, 0.0)

        frame = self.pipeline.get_frame()
        if frame is None:
            return

        h, w, _ = frame.shape
        # 初回 or サイズ変更時にテクスチャ作成
        if self.texture is None or self.texture.size != (w, h):
            self.texture = self.ctx.texture((w, h), 4)
            self.texture.build_mipmaps()

        self.texture.write(frame.tobytes())
        self.texture.use(location=0)
        self.prog['tex'] = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)
