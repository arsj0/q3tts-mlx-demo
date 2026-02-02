import gradio as gr
from mlx_audio.tts.utils import load_model
import numpy as np

# --- Configuration ---
MODEL_PATH_CV = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
MODEL_PATH_VD = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

SUPPORTED_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", 
    "Ryan", "Aiden", "Ono_Anna", "Sohee"
]

SUPPORTED_LANGUAGES = [
    "auto", "English", "Chinese", "Japanese", "Korean", 
    "German", "French", "Spanish", "Italian", "Portuguese", "Russian"
]

SPEAKER_INFO_MD = """
### Custom Voice Speakers Reference
| Speaker | Gender | Native Language / Accent | Tone / Nature |
| :--- | :--- | :--- | :--- |
| **Vivian** | Female | Chinese | Bright, slightly edgy, youthful. |
| **Serena** | Female | Chinese | Warm, gentle, soft. |
| **Uncle_Fu** | Male | Chinese | Seasoned, low, mellow timbre. |
| **Dylan** | Male | Chinese (Beijing) | Youthful, clear, natural Beijing dialect. |
| **Eric** | Male | Chinese (Sichuan) | Lively, slightly husky brightness, Sichuan dialect. |
| **Ryan** | Male | English | Dynamic, strong rhythmic drive. |
| **Aiden** | Male | English | Sunny, clear midrange, American accent. |
| **Ono_Anna** | Female | Japanese | Playful, light, nimble timbre. |
| **Sohee** | Female | Korean | Warm, rich emotion. |
"""

# --- Constants for Defaults ---
DEFAULT_TEXT_CV = "Hello! It's wonderful to meet you. I can speak many languages and express different emotions."
DEFAULT_TEXT_VD = "Hello. It's wonderful to meet you. This voice was created entirely from a text description."

DEFAULT_INSTRUCT_CV = "Warm and friendly, with a touch of excitement."
DEFAULT_INSTRUCT_VD = "A clear and elegant female voice with a gentle, sophisticated tone, speaking at a moderate pace with a touch of warmth."

# --- Global State ---
current_model = None
current_model_type = None # "Custom Voice" or "Voice Design"

# --- Logic ---

def load_model_if_needed(target_type):
    global current_model, current_model_type
    
    if current_model is not None and current_model_type == target_type:
        return current_model, f"Model {target_type} already loaded."

    print(f"Loading model: {target_type}...")
    # status_msg = f"Loading {target_type} model... Please wait."
    
    if target_type == "Custom Voice":
        current_model = load_model(MODEL_PATH_CV)
    else:
        current_model = load_model(MODEL_PATH_VD)
        
    current_model_type = target_type
    return current_model, f"Loaded {target_type} model."

def generate(mode, text, speaker, instruct, language, progress=gr.Progress()):
    if not text:
        yield None, "Please enter text to synthesize."
        return

    # Use progress bar and yield initial status
    progress(0.25, desc="Generating Audio...")
    yield None, "Generating..."

    try:
        model, msg = load_model_if_needed(mode)
    except Exception as e:
        yield None, f"Error loading model: {str(e)}"
        return

    progress(0.5, desc="Processing ...")

    try:
        # Normalize language
        lang_arg = language if language != "auto" else "auto"

        if mode == "Custom Voice":
            if speaker not in SUPPORTED_SPEAKERS:
                yield None, f"Invalid speaker: {speaker}"
                return
                
            results = list(model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=lang_arg,
                instruct=instruct
            ))
        else: # Voice Design
            results = list(model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=lang_arg
            ))
            
        if not results:
            yield None, "No audio generated."
            return
            
        progress(0.9, desc="Finalizing Audio...")
        result = results[0]
        audio_data = np.array(result.audio)
        
        # Format info string
        speaker_info = f", Speaker: {speaker}" if mode == "Custom Voice" else ""
        info_msg = f"Generation Completed. Model: {mode}{speaker_info}, Language: {language}"
        
        yield (result.sample_rate, audio_data), info_msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"Error during generation: {str(e)}"

# --- UI ---
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[ "sans-serif"],
).set(
    body_background_fill="#F8FAFC",
    body_text_color="#1E293B",
    block_background_fill="#FFFFFF",
    block_border_width="0px",
    block_shadow="0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
    button_primary_background_fill="#3B82F6",
    button_primary_background_fill_hover="#2563EB",
    button_primary_text_color="#FFFFFF",
    slider_color="#3B82F6",
    block_title_text_weight="600",
    block_label_text_weight="500",
    button_large_radius="24px",  # Pill shape for primary actions
    button_small_radius="12px",
    input_background_fill="#F1F5F9",
)

with gr.Blocks(theme=theme, title="Qwen3-TTS Demo") as demo:
    gr.Markdown("# Qwen3-TTS Demo (MLX)")
    gr.Markdown("Test the Qwen3-TTS models on MLX. Switch between **Custom Voice** and **Voice Design**.")
    
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(
                choices=["Custom Voice", "Voice Design"], 
                value="Custom Voice", 
                label="Model Mode",
                info="Custom Voice: Predefined speakers. Voice Design: Generate unique voices from description."
            )
            
            speaker = gr.Dropdown(
                choices=SUPPORTED_SPEAKERS, 
                value="Vivian", 
                label="Speaker", 
                interactive=True,
                visible=True
            )
            
            # Instruction label logic
            def update_visibility(selected_mode, current_text, current_instruct):
                is_custom = (selected_mode == "Custom Voice")
                speaker_update = gr.update(visible=is_custom)
                info_update = gr.update(visible=is_custom)
                
                # Determine new text value (only if it matches one of the defaults)
                new_text = current_text
                if current_text == DEFAULT_TEXT_CV or current_text == DEFAULT_TEXT_VD:
                    new_text = DEFAULT_TEXT_CV if is_custom else DEFAULT_TEXT_VD
                
                # Determine new instruct value (only if it matches one of the defaults)
                new_instruct = current_instruct
                if current_instruct == DEFAULT_INSTRUCT_CV or current_instruct == DEFAULT_INSTRUCT_VD:
                     new_instruct = DEFAULT_INSTRUCT_CV if is_custom else DEFAULT_INSTRUCT_VD

                if is_custom:
                    instruct_update = gr.update(
                        label="Style Instruction (Optional)", 
                        placeholder="e.g., 'Happy and excited', 'Sad', 'Whispering'",
                        value=new_instruct
                    )
                    text_update = gr.update(value=new_text)
                else:
                    instruct_update = gr.update(
                        label="Voice Description (Character/Timbre)", 
                        placeholder="e.g., 'A deep, resonant male voice, speaking slowly and with authority.'",
                        value=new_instruct
                    )
                    text_update = gr.update(value=new_text)
                
                return speaker_update, instruct_update, info_update, text_update

            text = gr.TextArea(
                label="Text to Synthesize", 
                placeholder="Enter text here...", 
                value=DEFAULT_TEXT_CV,
                lines=3
            )
            
            instruct = gr.TextArea(
                label="Style Instruction (Optional)", 
                placeholder="e.g., 'Happy and excited'", 
                value=DEFAULT_INSTRUCT_CV,
                lines=2
            )
            
            language = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES, 
                value="auto", 
                label="Language"
            )
            
            btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
            audio_out = gr.Audio(label="Generated Audio", type="numpy", interactive=False)
            speaker_info_md = gr.Markdown(SPEAKER_INFO_MD, visible=True)

    mode.change(fn=update_visibility, inputs=[mode, text, instruct], outputs=[speaker, instruct, speaker_info_md, text])
    
    btn.click(
        fn=generate, 
        inputs=[mode, text, speaker, instruct, language], 
        outputs=[audio_out, status]
    )

if __name__ == "__main__":
    demo.launch()
