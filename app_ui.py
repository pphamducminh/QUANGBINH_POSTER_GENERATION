# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.23.15", "pillow", "reportlab", "pymupdf"]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full", app_title="Quảng Bình · Poster Studio")


@app.cell
def _():
    import io
    import logging
    import re
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import marimo as mo
    from PIL import Image, ImageOps
    from template_manager import apply_template
    from model_selection import PipelineCache, local_models, resolve_model
    from core_model import GenerationSetupError, check_generation_runtime
    from endpoint_client import EndpointError, load_endpoint_config, generate_remote
    root = Path(__file__).resolve().parent
    endpoint_config = None
    try:
        endpoint_config = load_endpoint_config(root)
        if endpoint_config is None:
            check_generation_runtime()
        runtime_issue = None
    except EndpointError as _setup_error:
        runtime_issue = str(_setup_error)
    except GenerationSetupError:
        runtime_issue = "Chưa kết nối dịch vụ GPU. Cần hoàn tất triển khai Endpoint và cấu hình một lần; sau đó bạn chỉ cần mở app để tạo ảnh."

    get_bg, set_bg = mo.state(None)
    get_notice, set_notice = mo.state(None)
    get_used_model, set_used_model = mo.state(None)

    def _load_pipeline(source):
        # Keep the editor usable without GPU libraries or model authentication.
        # For gated models, authenticate on the host with `hf auth login`.
        from core_model import load_flux_pipeline
        return load_flux_pipeline(source)

    def _release_gpu():
        import torch
        torch.cuda.empty_cache()

    get_pipeline = PipelineCache(_load_pipeline, _release_gpu).get

    return (Image, ImageOps, Path, TemporaryDirectory, apply_template,
            get_bg, get_notice, get_pipeline, io, logging, mo, re, root,
            set_bg, set_notice, get_used_model, set_used_model, local_models, resolve_model, GenerationSetupError, runtime_issue, EndpointError, endpoint_config, generate_remote)


@app.cell
def _(mo):
    mo.Html("""
    <style>
      :root { --qb-ink: #173f38; --qb-muted: #65746d; --qb-line: #dce4dc; }
      body { background: #f7f8f3; }
      .qb-shell { max-width: 1380px; margin: 0 auto; padding: 16px 20px 40px;
        color: var(--qb-ink); font-family: system-ui, sans-serif; }
      .qb-top { display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 20px; border-bottom: 1px solid var(--qb-line); gap: 16px; }
      .qb-brand { font-size: 15px; font-weight: 750; letter-spacing: .03em; }
      .qb-brand span { display: inline-grid; place-items: center; background: #204e40;
        color: #fff; border-radius: 12px; width: 38px; height: 38px; margin-right: 10px; }
      .qb-badge { font-size: 12px; background: #e9eee3; padding: 8px 13px; border-radius: 30px; }
      .qb-hero { padding: 32px 0 26px; }
      .qb-eyebrow { font-size: 11px; font-weight: 750; letter-spacing: .16em; color: #6f8066; }
      .qb-hero h1 { font-size: clamp(28px, 4vw, 43px); line-height: 1.2;
        margin: 10px 0; letter-spacing: -.04em; font-weight: 650; }
      .qb-hero p, .qb-help { color: var(--qb-muted); font-size: 14px; line-height: 1.65; }
      .qb-grid { display: grid; grid-template-columns: minmax(300px, 410px) minmax(0, 1fr);
        align-items: start; gap: 24px; }
      .qb-card { background: #fff; border: 1px solid var(--qb-line); border-radius: 20px;
        padding: 24px; min-width: 0; box-shadow: 0 4px 20px #173f3804; }
      .qb-card h2 { font-size: 17px; font-weight: 650; margin: 0 0 8px; }
      .qb-step { color: #879b78; font-size: 12px; margin-right: 8px; }
      .qb-rule { height: 1px; background: var(--qb-line); margin: 24px 0; }
      .qb-stage { background: #eaf0e7; border-radius: 14px; padding: 24px;
        min-height: 440px; display: flex; align-items: center; justify-content: center; }
      .qb-stage img { max-height: 720px; width: auto; max-width: 100%; object-fit: contain;
        box-shadow: 0 14px 36px #18382a25; }
      .qb-empty { text-align: center; max-width: 300px; padding: 48px 12px; }
      .qb-empty-icon { font-size: 46px; color: #75916e; margin-bottom: 22px; }
      .qb-empty h3 { font-size: 21px; margin: 0 0 10px; }
      .qb-footer { margin-top: 24px; color: var(--qb-muted); font-size: 12px; }
      .qb-shell input, .qb-shell textarea, .qb-shell select { border-radius: 8px; }
      .qb-shell button { min-height: 40px; }
      @media (max-width: 820px) {
        .qb-grid { grid-template-columns: 1fr; }
        .qb-shell { padding: 8px 4px 24px; }
        .qb-card { padding: 18px; }
        .qb-stage { padding: 16px; min-height: 300px; }
        .qb-badge { display: none; }
      }
      @media (prefers-color-scheme: dark) {
        .qb-shell { color-scheme: light; }
      }
    </style>
    """)
    return


@app.cell
def _(endpoint_config, local_models, mo, root, runtime_issue):
    _model_options = {"Tự động · Ưu tiên model đã train": "auto"}
    for _folder in local_models(root):
        _label = _folder.rsplit("/", 1)[-1] if _folder != str(root) else "Thư mục gốc"
        _model_options[f"Đã train · {_label}"] = _folder
    _model_options["LoRA từ Hugging Face · cần GPU"] = "remote"
    if endpoint_config:
        _model_options = {"GPU tự động · Quảng Bình LoRA": "endpoint"}
    _default_model = next(iter(_model_options))
    source = mo.ui.radio(
        options=["Tạo ảnh bằng AI", "Tải ảnh của bạn", "Thử với ảnh mẫu"],
        value="Tạo ảnh bằng AI", label="Chọn nguồn ảnh", inline=False,
    )
    ai_form = mo.md("{model}\n\n{destination}\n\n{prompt}").batch(
        model=mo.ui.dropdown(options=_model_options,
            value=_default_model, label="Model tạo ảnh", full_width=True),
        destination=mo.ui.dropdown(
            options={"Phong Nha · Hang động": "pn_cave",
                     "Quang Phú · Đồi cát": "qp_dunes",
                     "Cảnh vật khác": ""},
            value="Phong Nha · Hang động", label="Bạn muốn khám phá đâu?", full_width=True),
        prompt=mo.ui.text_area(
            placeholder="Ví dụ: Ánh nắng xuyên qua cửa hang, dòng sông xanh ngọc, không khí yên bình…",
            label="Mô tả thêm ý tưởng", rows=4, full_width=True),
    ).form(submit_button_label="Tạo ảnh nền", bordered=False, clear_on_submit=False,
            submit_button_disabled=runtime_issue is not None, submit_button_tooltip=runtime_issue)
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"],
                        multiple=False, label="Chọn ảnh PNG, JPG hoặc WebP")
    sample_btn = mo.ui.run_button(label="Dùng ảnh mẫu Quang Phú")
    title_input = mo.ui.text(value="QUẢNG BÌNH", label="Tiêu đề", full_width=True, max_length=60)
    subtitle_input = mo.ui.text(value="Chạm vào vẻ đẹp nguyên sơ", label="Dòng giới thiệu", full_width=True, max_length=120)
    location_input = mo.ui.text(value="Quảng Bình · Việt Nam", label="Địa điểm", full_width=True, max_length=80)
    template_selector = mo.ui.dropdown(
        options={"Khám phá · Tạp chí du lịch": "template_1",
                 "Kỷ niệm · Bưu thiếp": "template_2",
                 "Hoài niệm · Phong cách retro": "template_3",
                 "Hành trình · Cẩm nang du lịch": "template_4",
                 "Chỉ lấy ảnh nền": "none"},
        value="Khám phá · Tạp chí du lịch", label="Bố cục poster", full_width=True)
    return (ai_form, location_input, sample_btn, source, subtitle_input,
            template_selector, title_input, upload)


@app.cell
def _(EndpointError, GenerationSetupError, ai_form, endpoint_config, generate_remote, get_pipeline, logging, mo, resolve_model, root, set_bg, set_notice, set_used_model):
    # Only submitted form values trigger model inference; editing text does not.
    if ai_form.value is not None:
        _values = ai_form.value
        _description = _values["prompt"].strip()
        if not _description and not _values["destination"]:
            set_notice(("warn", "Hãy chọn địa danh hoặc mô tả cảnh vật trước khi tạo ảnh."))
        else:
            try:
                _prompt = " ".join(filter(None, [
                    _values["destination"], _description,
                    "travel poster background, beautiful landscape, no text, high quality",
                ]))
                with mo.status.spinner(title="Đang tạo ảnh", subtitle="Lần đầu sau khi GPU nghỉ có thể cần vài phút…") as _spinner:
                    if endpoint_config:
                        _image = generate_remote(endpoint_config, _prompt,
                            on_status=lambda message: _spinner.update(subtitle=message))
                        _model_label = "GPU từ xa · kTon/quangbinh-flux-lora"
                    else:
                        from core_model import generate_poster_bg
                        _selected = resolve_model(root, _values["model"])
                        _image = generate_poster_bg(get_pipeline(_selected), _prompt, width=512, height=768, num_steps=28)
                        _model_label = _selected.label
                set_bg(_image)
                set_used_model(_model_label)
                set_notice(("success", "Ảnh đã sẵn sàng. Bạn có thể chỉnh nội dung và chọn bố cục bên dưới."))
            except (EndpointError, GenerationSetupError, FileNotFoundError) as _error:
                set_notice(("danger", str(_error)))
            except Exception:
                logging.exception("Poster background generation failed")
                set_notice(("danger", "Chưa tạo được ảnh. Hãy kiểm tra kết nối và Logs của dịch vụ tạo ảnh. Bạn vẫn có thể tải ảnh riêng hoặc dùng ảnh mẫu."))
    return


@app.cell
def _(Image, ImageOps, io, logging, set_bg, set_notice, set_used_model, upload):
    if upload.value:
        try:
            with Image.open(io.BytesIO(upload.value[0].contents)) as _uploaded:
                _uploaded = ImageOps.exif_transpose(_uploaded)
                _uploaded.thumbnail((2400, 3600))
                _image = _uploaded.convert("RGB").copy()
            set_bg(_image)
            set_used_model(None)
            set_notice(("success", "Đã nhận ảnh của bạn. Ảnh dọc sẽ phù hợp nhất với các mẫu poster."))
        except Exception:
            logging.exception("Poster image upload failed")
            set_notice(("danger", "Không đọc được ảnh này. Hãy chọn một ảnh PNG, JPG hoặc WebP hợp lệ."))
    return


@app.cell
def _(Image, logging, root, sample_btn, set_bg, set_notice, set_used_model):
    if sample_btn.value:
        try:
            with Image.open(root / "poster_quang_phu_final.png") as _sample:
                set_bg(_sample.convert("RGB").copy())
            set_used_model(None)
            set_notice(("info", "Đang dùng ảnh mẫu có sẵn. Bạn có thể thay ảnh bất cứ lúc nào."))
        except Exception:
            logging.exception("Sample image unavailable")
            set_notice(("warn", "Không tìm thấy ảnh mẫu. Hãy tải ảnh của bạn để tiếp tục."))
    return


@app.cell
def _(Image, Path, TemporaryDirectory, apply_template, get_bg, io,
      location_input, logging, mo, re, subtitle_input, template_selector, title_input):
    _bg = get_bg()
    downloads = mo.md("")
    if _bg is None:
        preview = mo.Html('''<div class="qb-empty"><div class="qb-empty-icon" aria-hidden="true">▧</div>
            <h3>Một hành trình, một tấm poster.</h3>
            <p class="qb-help">Tạo ảnh bằng AI, tải ảnh của bạn hoặc thử ảnh mẫu để bắt đầu.
            Poster sẽ xuất hiện tại đây.</p></div>''')
    else:
        _png = io.BytesIO()
        _bg.save(_png, format="PNG")
        _filename = re.sub(r"[^\w\-]+", "_", title_input.value.strip()).strip("_")[:60] or "Quang_Binh"
        _buttons = [mo.download(_png.getvalue(), filename=f"Anh_nen_{_filename}.png",
                                label="Tải ảnh nền PNG", mimetype="image/png")]
        preview = mo.image(_png.getvalue(), alt="Ảnh nền poster", width=440)
        if template_selector.value != "none":
            try:
                import pymupdf as fitz
                # Each render owns its temporary file; sessions never overwrite each other.
                with TemporaryDirectory(prefix="quangbinh-poster-") as _tmp:
                    _pdf_path = apply_template(template_selector.value, _bg,
                        title_input.value.strip() or "QUẢNG BÌNH",
                        subtitle_input.value.strip(), location_input.value.strip(),
                        output_path=str(Path(_tmp) / "poster.pdf"))
                    _pdf = Path(_pdf_path).read_bytes()
                with fitz.open(stream=_pdf, filetype="pdf") as _doc:
                    _page = _doc[0]
                    _scale = 1000 / max(_page.rect.width, _page.rect.height)
                    _render = _page.get_pixmap(matrix=fitz.Matrix(_scale, _scale)).tobytes("png")
                preview = mo.image(_render, alt="Bản xem trước poster với nội dung và bố cục đã chọn", width=440)
                _buttons.insert(0, mo.download(_pdf, filename=f"Poster_{_filename}.pdf",
                                  label="Tải poster PDF", mimetype="application/pdf"))
            except Exception:
                logging.exception("Poster rendering failed")
                preview = mo.vstack([mo.callout("Chưa dựng được bố cục này. Hãy thử mẫu khác; ảnh nền của bạn vẫn được giữ.", kind="warn"), preview])
        downloads = mo.hstack(_buttons, justify="center", wrap=True, gap=1)
    return downloads, preview


@app.cell
def _(ai_form, downloads, endpoint_config, get_bg, get_notice, get_used_model, location_input, mo, preview, resolve_model, root,
      sample_btn, source, subtitle_input, template_selector, title_input, upload, runtime_issue):
    _model_hint = mo.callout(
        "GPU tự động · Dùng LoRA Quảng Bình đã train. GPU tự bật khi tạo ảnh; lần đầu có thể chờ vài phút."
        if endpoint_config else "Tự động: " + resolve_model(root).label +
        ". Model được nạp trên máy đang chạy ứng dụng.", kind="info")
    _source_control = {"Tạo ảnh bằng AI": mo.vstack([_model_hint, mo.callout(runtime_issue, kind="warn") if runtime_issue else mo.md(""), ai_form]), "Tải ảnh của bạn": upload,
                       "Thử với ảnh mẫu": sample_btn}[source.value]
    _notice = get_notice()
    _message = mo.callout(_notice[1], kind=_notice[0]) if _notice else mo.md("")
    _image_controls = mo.vstack([source, _source_control, _message], gap=1.2)
    _text_controls = mo.vstack([template_selector, title_input, subtitle_input, location_input], gap=1.2)
    _used_model_note = mo.md("") if not get_used_model() else mo.callout("Ảnh hiện tại dùng: " + get_used_model(), kind="info")
    _status = "Sẵn sàng tải xuống" if get_bg() is not None else "Chờ ảnh của bạn"
    mo.md(f"""
    <div class="qb-shell">
      <header class="qb-top">
        <div class="qb-brand"><span aria-hidden="true">QB</span> QUẢNG BÌNH / POSTER STUDIO</div>
        <div class="qb-badge">Một chút sáng tạo. Một miền di sản.</div>
      </header>
      <section class="qb-hero">
        <div class="qb-eyebrow">MANG VẺ ĐẸP QUẢNG BÌNH VÀO TỪNG THIẾT KẾ</div>
        <h1>Chuyến đi của bạn. Poster của bạn.</h1>
        <p>Chọn một khung cảnh, kể câu chuyện của bạn và mang poster về.</p>
      </section>
      <div class="qb-grid">
        <section class="qb-card">
          <h2><span class="qb-step">01</span> Bắt đầu với một khung cảnh</h2>
          <p class="qb-help">Chọn cách bạn muốn tạo ảnh nền.</p>
          {_image_controls}
          <div class="qb-rule"></div>
          <h2><span class="qb-step">02</span> Thêm dấu ấn của bạn</h2>
          <p class="qb-help">Đổi mẫu hoặc sửa chữ để xem kết quả ngay, không cần tạo lại ảnh.</p>
          {_text_controls}
        </section>
        <section class="qb-card">
          <h2><span class="qb-step">03</span> Poster của bạn</h2>
          <p class="qb-help">{_status} · Bản xem trước cập nhật theo nội dung</p>
          {_used_model_note}
          <div class="qb-stage">{preview}</div>
          <div style="margin-top: 20px">{downloads}</div>
          <p class="qb-help" style="text-align:center;font-size:12px">PDF giữ nguyên bố cục và chữ · PNG là ảnh nền chưa thêm chữ</p>
        </section>
      </div>
      <footer class="qb-footer">QUẢNG BÌNH POSTER STUDIO · Cảm hứng từ thiên nhiên, sáng tạo bởi bạn.</footer>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
