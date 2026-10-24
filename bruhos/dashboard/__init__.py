"""Local, read-only console for saved bruhOS episodes."""
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import math
from pathlib import Path
from urllib.parse import urlsplit

from bruhos import __version__
from bruhos.trace import load_episode, verify_episode

WEB = Path(__file__).with_name('web')
PUBLIC = {'/': WEB / 'index.html'} | {
    '/' + p.relative_to(WEB).as_posix(): p
    for p in WEB.rglob('*') if p.is_file()
}
SUMMARY_FIELDS = (
    'episode_id', 'instruction', 'body', 'brain', 'started_at', 'success',
    'end_reason', 'recoveries', 'trace_root', 'n_steps',
)
TRACE_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError)


def status_data(root):
    candidates, episodes = [], []
    skipped = 0
    # Sort small headers first; read/hash step histories only for the visible page.
    for path in Path(root).glob('*/episode.json'):
        try:
            if path.parent.is_symlink() or path.is_symlink() or (path.parent / 'steps.jsonl').is_symlink():
                raise ValueError('Linked trace')
            header = json.loads(path.read_text())
            started = header['started_at']
            if type(started) not in (int, float) or not math.isfinite(started):
                raise ValueError('Invalid episode timestamp')
            json.dumps(header, allow_nan=False)
            candidates.append((started, path.parent))
        except TRACE_ERRORS:
            skipped += 1
    for _, directory in sorted(candidates, key=lambda item: item[0], reverse=True):
        try:
            episode = load_episode(directory)
            header = episode.header()
            json.dumps(header, allow_nan=False)
            summary = {key: header[key] for key in SUMMARY_FIELDS}
            episodes.append(dict(summary, verified=verify_episode(episode)))
        except TRACE_ERRORS:
            # Concurrent/incomplete records do not hide other readable episodes.
            skipped += 1
            continue
        if len(episodes) == 12:
            break
    return {'version': __version__, 'episodes': episodes, 'skipped': skipped}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/status':
            data, kind = json.dumps(status_data(self.root)).encode(), 'application/json'
        else:
            if path not in PUBLIC:
                self.send_error(404)
                return
            data = PUBLIC[path].read_bytes()
            kind = mimetypes.guess_type(str(PUBLIC[path]))[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def serve(root='traces', port=8765):
    with ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, root=Path(root))) as server:
        print(f'bruhOS trace console: http://127.0.0.1:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
