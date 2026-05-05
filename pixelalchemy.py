import argparse

from gst_pipeline import GStreamerPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend',
        choices=('vulkan', 'opengl'),
        default='vulkan',
        help='renderer backend to use',
    )
    parser.add_argument(
        '--video',
        default='video/sample.mp4',
        help='video file to process',
    )
    return parser.parse_args()


def run_opengl(pipeline):
    import moderngl_window as mglw

    from gl_renderer import GLRenderer

    GLRenderer.pipeline = pipeline
    mglw.run_window_config(GLRenderer)


def run_vulkan(pipeline):
    from vulkan_renderer import VulkanRenderer

    renderer = VulkanRenderer(pipeline)
    renderer.run()


def main():
    args = parse_args()

    pipeline = GStreamerPipeline(args.video)
    pipeline.start()

    if args.backend == 'opengl':
        run_opengl(pipeline)
    else:
        run_vulkan(pipeline)


if __name__ == '__main__':
    main()
