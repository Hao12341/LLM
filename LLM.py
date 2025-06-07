import gradio as gr
from transformers import pipeline, AutoTokenizer, M2M100ForConditionalGeneration
from langdetect import detect

# Carga modelo y tokenizer de traducción (M2M100)
translation_model_name = "facebook/m2m100_418M"
tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)

# Pipeline de resumen BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def translate(text, src_lang_code, tgt_lang_code):
    if src_lang_code not in tokenizer.lang_code_to_id:
        src_lang_code = "en"
    if tgt_lang_code not in tokenizer.lang_code_to_id:
        tgt_lang_code = "en"

    tokenizer.src_lang = src_lang_code
    encoded = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.get_lang_id(tgt_lang_code)
    generated_tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id,
        max_length=1024,
        num_beams=5,
        early_stopping=True
    )
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def greet(text):
    try:
        src_lang = detect(text)
    except:
        src_lang = "en"

    text_in_english = translate(text, src_lang, "en")

    max_words = 400
    max_tokens = 600
    result = summarizer(text_in_english, max_length=max_tokens, min_length=10, do_sample=False)
    resumen_ingles = result[0]["summary_text"]

    palabras = resumen_ingles.split()
    if len(palabras) > max_words:
        resumen_ingles = " ".join(palabras[:max_words])

    resumen_espanol = translate(resumen_ingles, "en", "es")

    return resumen_espanol, f"Resumen en español (texto original en: {src_lang})"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=10, max_length=5000, label="Texto para traducir y resumir"),
    outputs=[gr.Textbox(label="Resumen en español"), gr.Label(label="Idioma detectado")],
)

demo.launch()
