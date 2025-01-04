import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

model = YOLO("miku_detector_yolo11x.pt")

def predict(image, conf_thresh, progress=gr.Progress()):
    totalimg = len(image)
    valid_results = []
    for i, img in enumerate(progress.tqdm(image)):
        result = model(img, conf=conf_thresh)
        bgr_image = result[0].plot()
        rgb_image = cv2.cvtColor(np.array(bgr_image), cv2.COLOR_BGR2RGB)
        valid_results.append(Image.fromarray(rgb_image))
        progress(i + 1, desc=f"Processing image {i + 1}/{totalimg}")
    return valid_results

with gr.Blocks() as interface:
    gr.Markdown("# Miku Detector V1")
    gr.Markdown("#### Detects Miku, literally.")
    with gr.Accordion("模型评估指标 | In case you need it...", open=False):
        gr.Gallery(["assets/confusion_matrix.png",
                    "assets/confusion_matrix_normalized.png",
                    "assets/F1_curve.png",
                    "assets/P_curve.png",
                    "assets/PR_curve.png",
                    "assets/R_curve.png"], columns=6)
    with gr.Row():
        img_file = gr.Files(label="上传图像", file_types=["image"], height=300)
        conf_slider = gr.Slider(label="置信度阈值", minimum=0.1, maximum=1.0, step=0.01, value=0.8)
    with gr.Row():
        valid_output = gr.Gallery(type="pil", label="满足条件的结果", columns=5, interactive=False)
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Reset")

    submit_btn.click(
        fn=predict,
        inputs=[img_file, conf_slider],
        outputs=[valid_output],
        show_progress=True
    )

    clear_btn.click(
        fn=lambda: (None, None, 0.8),
        inputs=[],
        outputs=[img_file, valid_output, conf_slider]
    )


interface.launch()
