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
        self.edge_texture = None

        with open('shaders/edge.comp') as fcomp:
            self.comp = self.ctx.compute_shader(fcomp.read())

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
            self.edge_texture = self.ctx.texture((w, h), 1)
            self.texture.build_mipmaps()

        self.texture.write(frame.tobytes())

        self.texture.use(location=0)
        self.comp['src_tex'] = 0
        self.edge_texture.bind_to_image(1, read=False, write=True)
        gx, gy = (w + 15) // 16, (h + 15) // 16
        self.comp.run(gx, gy)

        self.ctx.memory_barrier()

        self.texture.use(location=0)
        self.edge_texture.use(location=1)
        self.prog['tex'] = 0
        self.prog['edge_tex'] = 1
        self.vao.render(moderngl.TRIANGLE_STRIP)
