import gc


class GenerationSetupError(RuntimeError):
    """An actionable runtime problem, safe to show in the app."""


def check_generation_runtime():
    try:
        import torch
    except ImportError as exc:
        raise GenerationSetupError(
            "Môi trường chạy app chưa có PyTorch. Hãy mở QuangBinh_Colab.ipynb "
            "trên Colab, bật GPU và chạy các ô thiết lập."
        ) from exc
    if not torch.cuda.is_available():
        raise GenerationSetupError(
            "Máy chạy app không có GPU NVIDIA/CUDA. Pipeline FLUX NF4 hiện tại "
            "không chạy trên GPU Apple của Mac. Hãy chạy QuangBinh_Colab.ipynb "
            "với GPU; model trên Hugging Face vẫn cần GPU để tạo ảnh."
        )
    import importlib.util
    missing = [name for name in ("diffusers", "transformers", "accelerate", "peft", "bitsandbytes", "sentencepiece")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise GenerationSetupError("Thiếu thư viện tạo ảnh: " + ", ".join(missing) +
                                   ". Hãy chạy ô cài thư viện trong notebook Colab.")


def load_flux_pipeline(lora_path, ckpt_id="black-forest-labs/FLUX.1-dev"):
    """
    Hàm này thực hiện lượng tử hóa, nạp base model, nạp LoRA 
    và dọn dẹp bộ nhớ. Chỉ gọi 1 lần khi khởi động UI.
    """
    check_generation_runtime()
    import torch
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import FluxTransformer2DModel, FluxPipeline
    from transformers import T5EncoderModel

    print("⏳ [1/4] Khởi tạo cấu hình lượng tử hóa (4-bit)...")
    bnb_4bit_compute_dtype = torch.float16

    nf4_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("⏳ [2/4] Tải Transformer và Text Encoder...")
    transformer = FluxTransformer2DModel.from_pretrained(
        ckpt_id, subfolder="transformer", quantization_config=nf4_config, torch_dtype=torch.float16
    )
    text_encoder = T5EncoderModel.from_pretrained(
        ckpt_id, subfolder="text_encoder_2", quantization_config=quant_config, torch_dtype=torch.float16
    )

    print("⏳ [3/4] Ráp Pipeline và tối ưu VRAM...")
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id, transformer=transformer, text_encoder_2=text_encoder, torch_dtype=bnb_4bit_compute_dtype
    )

    # Dọn dẹp RAM (Giữ nguyên logic cực tốt của bạn)
    del text_encoder, transformer
    gc.collect()
    torch.cuda.empty_cache()


    print(f"⏳ [4/4] Nạp trọng số LoRA từ: {lora_path}")
    pipeline.load_lora_weights(lora_path)
    # Offload inactive components to CPU instead of holding everything on GPU.
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_tiling()
    
    print("✅ Mô hình FLUX LoRA đã sẵn sàng trên GPU!")
    return pipeline


def generate_poster_bg(pipeline, user_prompt, width=512, height=768, num_steps=28, seed=None):
    """
    Hàm này nhận pipeline đã load và prompt từ người dùng,
    gọi GPU sinh ảnh trực tiếp bằng prompt đó.
    """
    import torch

    # 1. Trực tiếp sử dụng prompt người dùng truyền từ UI
    if user_prompt and user_prompt.strip():
        full_prompt = user_prompt.strip()
    else:
        # Prompt dự phòng nếu ô nhập liệu trên UI bị bỏ trống hoàn toàn
        full_prompt = "beautiful landscape, high quality, highly detailed"

    print(f"🚀 Bắt đầu vẽ ảnh (Steps: {num_steps}, Size: {width}x{height})...")
    print(f"📝 Prompt sử dụng: {full_prompt}")
    
    # 2. Thiết lập seed cố định nếu muốn gen lại ảnh y hệt, hoặc random nếu seed=None
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    # 3. Chạy suy luận
    image = pipeline(
        prompt=full_prompt, 
        num_inference_steps=num_steps,
        max_sequence_length=256, 
        guidance_scale=3.5, 
        height=height, 
        width=width, 
        generator=generator
    ).images[0]
    
    print("✅ Đã sinh ảnh xong!")
    return image
