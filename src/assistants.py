import openai
from io import BytesIO
import base64
from PIL import Image

client = openai.OpenAI()

def map_panel_to_description(panel: BytesIO, caption: str) -> str:
    """
    Map a single panel to its description using AI.

    Parameters:
        panel (BytesIO): The panel image.
        caption (str): The figure caption.

    Returns:
        str: The description for the panel.
    """
    panel_image = Image.open(panel)
    buffered = BytesIO()
    panel_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                    You will receive a text with the caption of a scientific figure. 
                    This figure will be generally composed of several panels. 
                    Extract the relevant part of the figure caption so that it matches the panel given as an image file. 
                    If a generic description of several panels is in place, return the generic and the specific descriptions for a given panel. 
                    Make sure that the information in the panel caption you return is enough to interpret the panel. ls 
                    For simplicity in post-processing begin the caption always with 'Panel X:' where X is the label of the panel in the figure.
                    
                    Output format:
                    ```
                    {
                        "panel_label": "X",
                        "panel_caption": "Description of the panel."
                    }
                    ```
                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{caption}"},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ],
        model="gpt-4o",
        n=1,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()
