import time

import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image


class ComputeApp(mglw.WindowConfig):
    gl_version = (4, 3)
    title = 'Gaussian Benchmark'
    window_size = (800, 600)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        w, h = self.window_size

        # 画像読み込み
        img = Image.open('image/sample.png').transpose(Image.FLIP_TOP_BOTTOM)
        img = img.resize(self.window_size)
        img_data = img.tobytes()

        # 入力
        self.src_texture = self.ctx.texture((w, h), 4, img_data)
        self.src_texture.bind_to_image(0, read=True, write=False)

        # 出力
        self.dst_texture = self.ctx.texture((w, h), 4)
        self.dst_texture.bind_to_image(1, read=False, write=True)

        # =========================
        # naive
        # =========================
        self.compute_naive = self.ctx.compute_shader('''
            #version 430
            layout(local_size_x = 16, local_size_y = 16) in;

            layout(rgba8, binding = 0) readonly uniform image2D src_img;
            layout(rgba8, binding = 1) writeonly uniform image2D dst_img;

            float kernel[5][5] = float[5][5](
                float[](1, 4, 6, 4, 1),
                float[](4,16,24,16,4),
                float[](6,24,36,24,6),
                float[](4,16,24,16,4),
                float[](1, 4, 6, 4, 1)
            );

            void main() {
                ivec2 id = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(src_img);

                if (id.x < 2 || id.y < 2 || id.x >= size.x-2 || id.y >= size.y-2) {
                    imageStore(dst_img, id, vec4(0,0,0,1));
                    return;
                }

                vec3 sum = vec3(0.0);
                float weight = 0.0;

                for (int y = -2; y <= 2; y++) {
                    for (int x = -2; x <= 2; x++) {
                        float w = kernel[y+2][x+2];
                        vec3 c = imageLoad(src_img, id + ivec2(x, y)).rgb;
                        sum += c * w;
                        weight += w;
                    }
                }

                sum /= weight;
                imageStore(dst_img, id, vec4(sum, 1.0));
            }
        ''')

        # =========================
        # shared（簡易版）
        # =========================
        self.compute_shared = self.ctx.compute_shader('''
            #version 430
            layout(local_size_x = 16, local_size_y = 16) in;

            layout(rgba8, binding = 0) readonly uniform image2D src_img;
            layout(rgba8, binding = 1) writeonly uniform image2D dst_img;

            shared vec3 tile[20][20];

            float kernel[5][5] = float[5][5](
                float[](1, 4, 6, 4, 1),
                float[](4,16,24,16,4),
                float[](6,24,36,24,6),
                float[](4,16,24,16,4),
                float[](1, 4, 6, 4, 1)
            );

            void main() {
                ivec2 id = ivec2(gl_GlobalInvocationID.xy);
                ivec2 lid = ivec2(gl_LocalInvocationID.xy);
                ivec2 size = imageSize(src_img);

                ivec2 tpos = lid + ivec2(2, 2);

                // 中央
                tile[tpos.y][tpos.x] = imageLoad(src_img, id).rgb;

                // 周囲（簡易）
                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        ivec2 p = id + ivec2(dx, dy);
                        ivec2 tp = tpos + ivec2(dx, dy);

                        if (tp.x >= 0 && tp.x < 20 && tp.y >= 0 && tp.y < 20) {
                            tile[tp.y][tp.x] = imageLoad(src_img, p).rgb;
                        }
                    }
                }

                barrier();

                if (id.x < 2 || id.y < 2 || id.x >= size.x-2 || id.y >= size.y-2) {
                    imageStore(dst_img, id, vec4(0,0,0,1));
                    return;
                }

                vec3 sum = vec3(0.0);
                float weight = 0.0;

                for (int y = -2; y <= 2; y++) {
                    for (int x = -2; x <= 2; x++) {
                        float w = kernel[y+2][x+2];
                        sum += tile[tpos.y + y][tpos.x + x] * w;
                        weight += w;
                    }
                }

                sum /= weight;
                imageStore(dst_img, id, vec4(sum, 1.0));
            }
        ''')

        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
            1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 uv;
                void main() {
                    uv = in_uv;
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec2 uv;
                out vec4 fragColor;
                uniform sampler2D tex;
                void main() {
                    fragColor = texture(tex, uv);
                }
            '''
        )

        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f', 'in_pos', 'in_uv')]
        )

        self.dst_texture.use()
        self.prog['tex'] = 0

        self.done = False

    def benchmark(self, compute, gx, gy, n=30):
        for _ in range(5):
            compute.run(group_x=gx, group_y=gy)
        self.ctx.finish()

        t0 = time.time()
        for _ in range(n):
            compute.run(group_x=gx, group_y=gy)
        self.ctx.finish()
        t1 = time.time()

        return (t1 - t0) / n

    def on_render(self, time, frame_time):
        w, h = self.window_size
        gx = (w + 15) // 16
        gy = (h + 15) // 16

        if not self.done:
            self.done = True

            t1 = self.benchmark(self.compute_naive, gx, gy)
            t2 = self.benchmark(self.compute_shared, gx, gy)

            print("naive :", t1)
            print("shared:", t2)
            print("diff  :", t1 - t2)

        self.compute_shared.run(group_x=gx, group_y=gy)

        self.ctx.clear(0, 0, 0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


def main():
    mglw.run_window_config(ComputeApp)


if __name__ == '__main__':
    main()
