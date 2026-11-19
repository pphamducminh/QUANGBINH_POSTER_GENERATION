"""Server-side client for the project's protected Hugging Face GPU endpoint."""
import base64
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image


class EndpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class EndpointConfig:
    url: str
    token: str = field(repr=False)


def load_endpoint_config(root):
    config_path = Path(root) / '.config' / 'endpoint.json'
    try:
        values = json.loads(config_path.read_text()) if config_path.exists() else {}
        url = os.environ.get('HF_ENDPOINT_URL', values.get('url', '')).strip()
        if not url:
            return None
        token = os.environ.get('HF_ENDPOINT_TOKEN') or os.environ.get('HF_TOKEN') or ''
        if not token and values.get('token_file'):
            token_path = Path(values['token_file'])
            if not token_path.is_absolute():
                token_path = config_path.parent / token_path
            token = token_path.read_text().strip()
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token() or ''
            except ImportError:
                pass
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        raise EndpointError('Không đọc được cấu hình Endpoint. Chạy lại scripts/configure_endpoint.py.') from exc
    validate_endpoint_url(url)
    if not token:
        raise EndpointError('Endpoint chưa có token xác thực. Chạy scripts/configure_endpoint.py để cấu hình một lần.')
    return EndpointConfig(url, token)


def validate_endpoint_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname
            or not parsed.hostname.endswith('.endpoints.huggingface.cloud')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.port not in (None, 443)):
        raise EndpointError('URL phải là địa chỉ HTTPS của Hugging Face Inference Endpoint.')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a bearer token to a redirected destination.


def generate_remote(config, prompt, *, seed=None, timeout=900, on_status=None,
                    opener=None, clock=time.monotonic, sleep=time.sleep):
    validate_endpoint_url(config.url)
    if not prompt.strip():
        raise EndpointError('Hãy nhập mô tả ảnh.')
    opener = opener or urllib.request.build_opener(_NoRedirect()).open
    deadline = clock() + timeout
    payload = json.dumps({'inputs': prompt, 'parameters': {
        'width': 512, 'height': 768, 'num_inference_steps': 28, 'seed': seed,
    }}).encode()
    if on_status:
        on_status('Đang gửi yêu cầu tạo ảnh. Nếu GPU đang nghỉ, lần đầu có thể chờ vài phút…')
    while clock() < deadline:
        remaining = deadline - clock()
        req = urllib.request.Request(config.url, data=payload, headers={
            'Authorization': f'Bearer {config.token}', 'Content-Type': 'application/json',
            'Accept': 'application/json', 'X-Scale-Up-Timeout': str(max(1, min(600, int(remaining)))),
        })
        try:
            with opener(req, timeout=max(1, remaining)) as response:
                raw = response.read(24 * 1024 * 1024 + 1)
                if len(raw) > 24 * 1024 * 1024:
                    raise EndpointError('Ảnh trả về vượt quá giới hạn dung lượng.')
                result = json.loads(raw)
            image_bytes = base64.b64decode(result['image'], validate=True)
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.size != (512, 768):
                    raise EndpointError('Endpoint trả về ảnh không đúng kích thước đã yêu cầu.')
                return image.convert('RGB').copy()
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                if on_status:
                    on_status('GPU đang khởi động hoặc tạm thời chưa sẵn sàng. Ứng dụng sẽ tự chờ…')
                sleep(min(5, max(0, deadline - clock())))
                continue
            messages = {
                401: 'Token Hugging Face không hợp lệ hoặc đã hết hạn.',
                403: 'Token chưa có quyền gọi Endpoint này.',
                404: 'Không tìm thấy Endpoint. Kiểm tra URL và trạng thái triển khai.',
                429: 'Endpoint đang bận hoặc vượt giới hạn. Hãy thử lại sau.',
                504: 'GPU chưa trả ảnh kịp thời. Kiểm tra trạng thái trước khi gửi lại để tránh tạo ảnh trùng.',
            }
            raise EndpointError(messages.get(exc.code, f'Dịch vụ tạo ảnh trả lỗi HTTP {exc.code}. Kiểm tra Logs của Endpoint.')) from None
        except (TimeoutError, urllib.error.URLError):
            raise EndpointError('Kết nối bị gián đoạn hoặc quá thời gian chờ. Yêu cầu có thể vẫn đang xử lý; kiểm tra Endpoint trước khi gửi lại.') from None
        except (ValueError, KeyError, TypeError, OSError):
            raise EndpointError('Endpoint trả dữ liệu ảnh không hợp lệ. Kiểm tra phiên bản handler đã triển khai.') from None
    raise EndpointError('GPU chưa sẵn sàng sau thời gian chờ. Hãy kiểm tra trạng thái và Logs của Endpoint.')
