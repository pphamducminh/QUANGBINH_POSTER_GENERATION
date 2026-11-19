"""Build a self-contained, interactive molab app; no remote server or keepalive."""
import base64
import io
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
files = [ROOT / name for name in ('core_model.py', 'model_selection.py', 'endpoint_client.py', 'template_manager.py')]
files += sorted((ROOT / 'templates').glob('*.py'))
# Fonts are downloaded on first use to keep the notebook import small.
files.append(ROOT / 'poster_quang_phu_final.png')
archive = io.BytesIO()
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in files:
        z.write(path, path.relative_to(ROOT))
payload = base64.b64encode(archive.getvalue()).decode()
source = (ROOT / 'app_ui.py').read_text()
source = source.replace('app_title="Quảng Bình · Poster Studio"', 'app_title="Quảng Bình · GPU miễn phí trên molab"')
bootstrap = '''    import base64 as _base64
    import importlib as _importlib
    import importlib.util as _import_util
    import os as _os
    import subprocess as _subprocess
    import sys as _sys
    import zipfile as _zipfile
    _runtime = Path.cwd() / ".quangbinh_runtime"
    if Path.cwd().name == ".quangbinh_runtime":
        _runtime = Path.cwd()
    _runtime.mkdir(exist_ok=True)
    _payload = "PAYLOAD_VALUE"
    with _zipfile.ZipFile(io.BytesIO(_base64.b64decode(_payload))) as _archive:
        for _name in _archive.namelist():
            if not (_runtime / _name).resolve().is_relative_to(_runtime.resolve()):
                raise ValueError("Invalid bundled path")
        _archive.extractall(_runtime)
    if str(_runtime) not in _sys.path:
        _sys.path.insert(0, str(_runtime))
    _os.chdir(_runtime)
    import urllib.request as _urlrequest
    _font_urls = {
        "NotoSans-Regular.ttf": "https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        "NotoSans-Bold.ttf": "https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
        "Anton-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
        "GreatVibes-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/greatvibes/GreatVibes-Regular.ttf",
    }
    (_runtime / "fonts").mkdir(exist_ok=True)
    for _font_name, _font_url in _font_urls.items():
        _font_path = _runtime / "fonts" / _font_name
        if not _font_path.exists():
            with _urlrequest.urlopen(_font_url, timeout=60) as _response:
                _font_data = _response.read()
            _font_temp = _font_path.with_suffix(".tmp")
            _font_temp.write_bytes(_font_data)
            _font_temp.replace(_font_path)
    _packages = {"PIL": "pillow", "reportlab": "reportlab", "pymupdf": "pymupdf", "huggingface_hub": "huggingface-hub"}
    import torch as _torch
    if _torch.cuda.is_available():
        _packages.update({"diffusers": "diffusers>=0.39,<0.44", "transformers": "transformers>=5.14,<6",
            "accelerate": "accelerate>=1.14,<2", "peft": "peft>=0.19,<0.22",
            "bitsandbytes": "bitsandbytes>=0.49.2", "sentencepiece": "sentencepiece", "safetensors": "safetensors"})
    _missing = [_package for _module, _package in _packages.items() if _import_util.find_spec(_module) is None]
    if _missing:
        with mo.status.spinner(title="Đang chuẩn bị thư viện cho phiên này…"):
            _subprocess.run([_sys.executable, "-m", "pip", "install", "--quiet", *_missing], check=True)
        _importlib.invalidate_caches()
'''.replace('PAYLOAD_VALUE', payload)
source = source.replace('    from PIL import Image, ImageOps', bootstrap+'    from PIL import Image, ImageOps',1)
source = source.replace('root = Path(__file__).resolve().parent','root = _runtime')
source = source.replace('endpoint_config = load_endpoint_config(root)', 'endpoint_config = None  # Free molab build never invokes a paid Endpoint.')
source = source.replace('runtime_issue = "Chưa kết nối dịch vụ GPU. Cần hoàn tất triển khai Endpoint và cấu hình một lần; sau đó bạn chỉ cần mở app để tạo ảnh."', 'runtime_issue = "Bật GPU trong nút thông số máy ở đầu trang molab, rồi chạy lại notebook. Phiên miễn phí có giới hạn thời gian."')
source = source.replace('        return load_flux_pipeline(source)', '''        from huggingface_hub import hf_hub_download
        try:
            hf_hub_download("black-forest-labs/FLUX.1-dev", "model_index.json")
        except Exception as error:
            raise GenerationSetupError("Chưa truy cập được FLUX.1-dev. Đăng nhập bằng token Read phía trên và chấp nhận quyền truy cập model trên Hugging Face; kiểm tra kết nối nếu đã cấp quyền.") from error
        return load_flux_pipeline(source)''')
# An explicit session-only login form; no embedded credentials or token persistence uploads.
login_cells = '''
@app.cell
def _(mo):
    from huggingface_hub import get_token, login
    get_auth_ready, set_auth_ready = mo.state(bool(get_token()))
    hf_login = mo.ui.text(kind="password", label="Hugging Face Read token", full_width=True).form(
        submit_button_label="Kết nối Hugging Face", clear_on_submit=True, bordered=False)
    return get_auth_ready, set_auth_ready, hf_login, login


@app.cell
def _(hf_login, login, set_auth_ready, mo):
    if hf_login.value:
        try:
            login(token=hf_login.value.strip(), add_to_git_credential=False)
            set_auth_ready(True)
        except Exception:
            mo.output.append(mo.callout("Token chưa hợp lệ hoặc kết nối lỗi. Hãy kiểm tra rồi thử lại.", kind="warn"))
    return

'''
source = source.replace('if __name__ == "__main__":', login_cells+'\nif __name__ == "__main__":')
source = source.replace('sample_btn, source, subtitle_input, template_selector, title_input, upload, runtime_issue):', 'sample_btn, source, subtitle_input, template_selector, title_input, upload, runtime_issue, get_auth_ready, hf_login):')
source = source.replace('    _image_controls = mo.vstack([source, _source_control, _message], gap=1.2)', '''    _auth_panel = mo.callout("Đã có thông tin đăng nhập Hugging Face trong phiên này.", kind="success") if get_auth_ready() else mo.vstack([
        mo.md("**Kết nối Hugging Face** — dùng token Read của tài khoản đã được cấp quyền [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). Token không được ghi vào mã notebook. Phiên mới có thể cần nhập lại."), hf_login])
    _image_controls = mo.vstack([_auth_panel, source, _source_control, _message], gap=1.2)''')
source = source.replace('Một chút sáng tạo. Một miền di sản.', 'molab · GPU miễn phí theo phiên')
source = source.replace('Cảm hứng từ thiên nhiên, sáng tạo bởi bạn.', 'Không cần train lại · Model được nạp một lần mỗi phiên · Phiên có thể hết hạn.')
(ROOT / 'QuangBinh_Molab.py').write_text(source)
print('Built QuangBinh_Molab.py with',len(files),'bundled files; no credentials or model weights.')
