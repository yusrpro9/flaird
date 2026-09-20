import os
import re

import gradio as gr

import spaces
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = os.environ.get("MODEL_ID")
max_length = int(os.environ.get("MAX_LENGTH", 512))

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = (
    AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    .eval()
    .to(device)
)


def preprocess(text: str) -> str:
    EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")
    PHONE_PATTERN = re.compile(
        r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}"
    )
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(USER_MENTION_PATTERN, "[USER]", text)
    text = re.sub(PHONE_PATTERN, " [PHONE]", text).replace("  [PHONE]", " [PHONE]")
    return text.strip()


@spaces.GPU
def predict(text: str) -> str:

    text = preprocess(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs["forensic_features"] = model.extract_forensic_features([text], return_tensors=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_fusion_states=True)

    logits = outputs.logits.cpu()[0]
    score = torch.sigmoid(logits).numpy()

    return f"machine score: {score}"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Text(
        value="""
Duke Ellington, a titan of jazz, revolutionized the genre through his innovative compositions, showcasing a remarkable ability to integrate voice and instrumental music. Among the notable figures who contributed to this artistic symphony was Ivie Anderson, whose scat singing mirrored the improvisational prowess of musicians like Nanton. In "Ring Dem Bells," Ellington ingeniously interweaves scat singing as a dialogue with saxophones, creating a dynamic call and response that underscores his vision of music as a fluid conversation between voices and instruments.

Ellington consistently emphasized the voice as an instrument of equal importance to traditional brass and woodwinds, orchestrating his compositions with a keen ear for vocal qualities. This approach is vividly demonstrated in "Mood Indigo," where his orchestration skills shine through non-traditional chord arrangements, transforming the piece into an auditory tapestry of mood and color. It is not merely the notes that define Ellington's genius; rather, it is the way he orchestrates these elements, drawing from the unique talents of band members like Nanton, Hodges, and Williams, to create a symphony where each voice resonates with authenticity and purpose.

The composition "Dusk" exemplifies Ellington's adeptness at capturing mood through orchestration, exploring tone inversions in a manner that surpasses the treatment in "Mood Indigo." Here, he delves into the emotional depths, using music to paint a vivid picture of dusk, where shadows lengthen and the world slows. This piece, alongside his faster-paced big band classics, demonstrates his versatile orchestration skills, capable of evoking warmth and romance while maintaining the vitality and energy associated with his signature style.

In "Daybreak Express," Ellington employs a metaphor akin to a speeding train, utilizing orchestration to tell a thematic story of movement and progress. This vivid imagery highlights his ability to infuse music with narrative power, transforming a simple composition into a journey underscored by the crescendo and diminuendo of his adventurous harmonies. Furthermore, the cohesion and skill evident in his larger orchestras, as seen in works like "Harlem Air Shaft," speak volumes about his mastery over ensemble playing. Ellington's legacy lies not only in his vast repertoire but in how he redefined the boundaries of jazz, crafting a musical world where innovation and tradition harmoniously coexist.
"""
    ),
    outputs=gr.Text(),
)
demo.launch()
