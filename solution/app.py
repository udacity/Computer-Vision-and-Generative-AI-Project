import gradio as gr
import numpy as np
from PIL import Image

input_points = []

def generate_app(get_processed_inputs, inpaint):
    HEIGHT = 512
    WIDTH = 512

    def get_points(img, evt: gr.SelectData):
        
        global input_points
        
        x = evt.index[0]
        y = evt.index[1]

        input_points.append([x, y])

        gr.Info(f"Added point {x}, {y}")


    def run_sam(raw_image):
        global input_points
        
        try:

            if len(input_points) == 0:
                raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")

            gr.Info(f"Received {len(input_points)} points, processing with SAM")
            mask = get_processed_inputs(raw_image, [input_points])

            input_points = []

            res_mask = Image.fromarray(mask).resize((HEIGHT, WIDTH))

            return res_mask
        except Exception as e:
            raise gr.Error(str(e))


    def run(raw_image, mask, prompt, negative_prompt, cfg, seed):

        gr.Info("Inpainting... (this will take up to a few minutes)")
        try:
            inpainted = inpaint(raw_image, np.array(mask), prompt, negative_prompt, seed, cfg)
        except Exception as e:
            raise gr.Error(str(e))

        return inpainted


    with gr.Blocks() as demo:

        gr.Markdown(
        """
        # Background swap
        1. Select an image by clicking on the first canvas. A square image would work best.
        2. Click a few times *slowly* on different points of the subject 
           you want to keep. For example, in the Mona Lisa example, click multiple times on different areas of Mona 
           Lisa's skin.
        3. Click on "run SAM". This will show you the mask that will be used. If you don't like the mask, 
           retry again point 2 (note: points are NOT kept from the previous try)
        4. Write a prompt (and optionally a negative prompt) for what you want to generate as the background of your 
           subject. Adjust the CFG scale and the seed if needed.
        5. Click on "run inpaint". If you are not happy with the result, change your prompts and/or the 
           settings (CFG scale, random seed) and click "run inpaint" again.

        > NOTE: the generation of the background can take up to a couple of minutes. Be patient!

        # EXAMPLES
        Scroll down to see a few examples. Click on an example and the image and the prompts will be filled for you. 
        Note however that you still need to do step 2, 3, and 5.
        """)

        with gr.Row():
            input_img = gr.Image(
                label="Input", 
                interactive=True, 
                type='pil',
                height=HEIGHT,
                width=WIDTH
            )
            input_img.select(get_points, inputs=[input_img])

            sam_mask = gr.Image(
                label="SAM result",
                interactive=False,
                type='pil',
                height=HEIGHT,
                width=WIDTH,
                elem_id="sam_mask"
            )

            result = gr.Image(
                label="Output",
                interactive=False,
                type='pil',
                height=HEIGHT,
                width=WIDTH,
                elem_id="output_image"
            )

        with gr.Row():
            cfg = gr.Slider(
                label="Classifier-Free Guidance Scale", minimum=0.0, maximum=20.0, value=7, step=0.05
            )
            random_seed = gr.Number(
                label="Random seed", 
                value=74294536, 
                precision=0
            )

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt for infill"
            )
            neg_prompt = gr.Textbox(
                label="Negative prompt"
            )
            submit_sam = gr.Button(value="run SAM")
            submit_inpaint = gr.Button(value="run inpaint")

        with gr.Row():
            examples = gr.Examples(
                [
                    [
                        "car.png", 
                        "a car driving on the Mars. Studio lights, 1970s", 
                        "artifacts, low quality, distortion",
                        74294536
                    ],
                    [
                        "dragon.jpeg",
                        "a dragon in a medieval village",
                        "artifacts, low quality, distortion",
                        97
                    ],
                    [
                        "monalisa.png",
                        "a fantasy landscape with flying dragons",
                        "artifacts, low quality, distortion",
                        97
                    ]
                ],
                inputs=[
                    input_img,
                    prompt,
                    neg_prompt,
                    random_seed
                ]

            )

        submit_sam.click(
            fn=run_sam,
            inputs=[input_img],
            outputs=[sam_mask]
        )

        submit_inpaint.click(
            fn=run, 
            inputs=[
                input_img, 
                sam_mask,
                prompt, 
                neg_prompt,
                cfg,
                random_seed
            ], 
            outputs=[result]
        )

    demo.queue(max_size=1).launch(share=True, debug=True)
    
    return demo