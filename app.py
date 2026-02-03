import gradio as gr
from mlx_audio.tts.utils import load_model
import numpy as np

# --- Configuration ---
MODEL_PATH_CV = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
MODEL_PATH_VD = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
MODEL_PATH_BASE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

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
DEFAULT_TEXT_CLONE = "Hello. It's wonderful to meet you. This voice is a clone of the reference audio you provided."

DEFAULT_INSTRUCT_CV = "Warm and friendly, with a touch of excitement."
DEFAULT_INSTRUCT_VD = "A clear and elegant female voice with a gentle, sophisticated tone, speaking at a moderate pace with a touch of warmth."

# --- Global State ---
current_model = None
current_model_type = None # "Custom Voice", "Voice Design", or "Voice Clone"

# --- Logic ---

def load_model_if_needed(target_type):
    global current_model, current_model_type
    
    if current_model is not None and current_model_type == target_type:
        return current_model, f"Model {target_type} already loaded."

    print(f"Loading model: {target_type}...")
    # status_msg = f"Loading {target_type} model... Please wait."
    
    if target_type == "Custom Voice":
        current_model = load_model(MODEL_PATH_CV)
    elif target_type == "Voice Design":
        current_model = load_model(MODEL_PATH_VD)
    else: # Voice Clone
        current_model = load_model(MODEL_PATH_BASE)
        
    current_model_type = target_type
    return current_model, f"Loaded {target_type} model."

def generate(mode, text, speaker, instruct, ref_audio, ref_text, language, progress=gr.Progress()):
    if not text:
        yield None, "<span style='color: red'>Please enter text to synthesize.</span>"
        return

    # Use progress bar and yield initial status
    progress(0.25, desc="Generating Audio...")
    yield None, "Generating..."

    try:
        model, msg = load_model_if_needed(mode)
    except Exception as e:
        yield None, f"<span style='color: red'>Error loading model: {str(e)}</span>"
        return

    progress(0.5, desc="Processing ...")

    try:
        # Normalize language
        lang_arg = language if language != "auto" else "auto"

        if mode == "Custom Voice":
            if speaker not in SUPPORTED_SPEAKERS:
                yield None, f"<span style='color: red'>Invalid speaker: {speaker}</span>"
                return
                
            results = list(model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=lang_arg,
                instruct=instruct
            ))
        elif mode == "Voice Design":
            results = list(model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=lang_arg
            ))
        else: # Voice Clone
            if not ref_audio:
                 yield None, "<span style='color: red'>Reference audio is required for Voice Clone.</span>"
                 return
                 
            # If ref_text is empty, treat as None
            ref_text_arg = ref_text if ref_text and ref_text.strip() else None

            results = list(model.generate(
                text=text,
                ref_audio=ref_audio, # Gradio passes the filepath
                ref_text=ref_text_arg,
                language=lang_arg
            ))
            
        if not results:
            yield None, "<span style='color: red'>No audio generated.</span>"
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
        yield None, f"<span style='color: red'>Error during generation: {str(e)}</span>"

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

with gr.Blocks(theme=theme, title="Qwen3-TTS MLX Demo") as demo:
    gr.Markdown("## Qwen3-TTS MLX Demo")
    gr.Markdown("Test the Qwen3-TTS models on MLX. Switch between **Custom Voice**, **Voice Design**, and **Voice Clone**.")
    
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(
                choices=["Custom Voice", "Voice Design", "Voice Clone"], 
                value="Custom Voice", 
                label="Model Mode",
                info="• Custom Voice: Predefined speakers.\n• Voice Design: Create custom voices using natural language descriptions.\n• Voice Clone: Clone any voice from a reference audio."
            )
            
            speaker = gr.Dropdown(
                choices=SUPPORTED_SPEAKERS, 
                value="Vivian", 
                label="Speaker", 
                interactive=True,
                visible=True
            )
            
            # Voice Clone Inputs
            ref_audio = gr.Audio(
                label="Reference Audio", 
                type="filepath", 
                visible=False
            )
            
            ref_text = gr.Textbox(
                label="Reference Text (Optional)", 
                placeholder="Transcript of the reference audio to improve cloning quality...", 
                visible=False
            )
            
            # Instruction label logic
            def update_visibility(selected_mode, current_text, current_instruct):
                is_custom = (selected_mode == "Custom Voice")
                is_design = (selected_mode == "Voice Design")
                is_clone = (selected_mode == "Voice Clone")
                
                speaker_update = gr.update(visible=is_custom)
                info_update = gr.update(visible=is_custom)
                
                # Clone inputs visibility
                ref_audio_update = gr.update(visible=is_clone)
                ref_text_update = gr.update(visible=is_clone)
                
                # Determine new text value 
                new_text = current_text
                if current_text in [DEFAULT_TEXT_CV, DEFAULT_TEXT_VD, DEFAULT_TEXT_CLONE]:
                    if is_custom: new_text = DEFAULT_TEXT_CV
                    elif is_design: new_text = DEFAULT_TEXT_VD
                    elif is_clone: new_text = DEFAULT_TEXT_CLONE
                
                # Determine new instruct value and visibility
                # Instruct is hidden in Clone mode
                new_instruct = current_instruct
                instruct_vis = not is_clone
                
                if current_instruct in [DEFAULT_INSTRUCT_CV, DEFAULT_INSTRUCT_VD]:
                     if is_custom: new_instruct = DEFAULT_INSTRUCT_CV
                     elif is_design: new_instruct = DEFAULT_INSTRUCT_VD

                if is_custom:
                    instruct_update = gr.update(
                        label="Style Instruction (Optional)", 
                        placeholder="e.g., 'Happy and excited', 'Sad', 'Whispering'",
                        value=new_instruct,
                        visible=True
                    )
                elif is_design:
                    instruct_update = gr.update(
                        label="Voice Description (Character/Timbre)", 
                        placeholder="e.g., 'A deep, resonant male voice, speaking slowly and with authority.'",
                        value=new_instruct,
                        visible=True
                    )
                else: # Clone
                    instruct_update = gr.update(visible=False)

                text_update = gr.update(value=new_text)
                
                return speaker_update, instruct_update, info_update, text_update, ref_audio_update, ref_text_update

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
            gr.Markdown("### Status")
            status = gr.Markdown(value="Ready")
            audio_out = gr.Audio(label="Generated Audio", type="numpy", interactive=False)
            speaker_info_md = gr.Markdown(SPEAKER_INFO_MD, visible=True)

    mode.change(
        fn=update_visibility, 
        inputs=[mode, text, instruct], 
        outputs=[speaker, instruct, speaker_info_md, text, ref_audio, ref_text]
    )
    
    btn.click(
        fn=generate, 
        inputs=[mode, text, speaker, instruct, ref_audio, ref_text, language], 
        outputs=[audio_out, status]
    )

if __name__ == "__main__":
    demo.launch()
