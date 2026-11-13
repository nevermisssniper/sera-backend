import hashlib
import hmac
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))
from generation_security import signed_callback, callback_config
import predict

JOB = 'a1f9aeba-bb00-4a0d-8eaa-bf3112095ae4'
IMAGE = 'b1f9aeba-bb00-4a0d-8eaa-bf3112095ae4'


def test_generation_requires_auth_and_rejects_callback_override(monkeypatch):
    task = Mock()
    task.apply_async.return_value.id = JOB
    monkeypatch.setitem(sys.modules, 'tasks', SimpleNamespace(generate_images_task=task))
    sys.modules.pop('main', None)
    main = importlib.import_module('main')
    monkeypatch.setenv('GENERATION_SERVICE_TOKEN', 'test-service-token')
    monkeypatch.setenv('GENERATION_CALLBACK_SECRET', 'test-callback-secret')
    monkeypatch.setenv('GENERATION_CALLBACK_URL', 'https://example.com/api/generate-callback')
    client = TestClient(main.app)
    payload = {'userId': 'user', 'jobId': JOB, 'instanceImages': [IMAGE]}
    assert client.post('/generate', json=payload).status_code == 401
    task.apply_async.assert_not_called()
    headers = {'Authorization': 'Bearer test-service-token'}
    assert client.post('/generate', json={**payload, 'callbackUrl': 'https://attacker.example'}, headers=headers).status_code == 422
    assert client.post('/generate', json=payload, headers=headers).status_code == 200
    assert task.apply_async.call_args.kwargs['args'][0]['jobId'] == JOB
    monkeypatch.delenv('GENERATION_SERVICE_TOKEN')
    assert client.post('/generate', json=payload, headers=headers).status_code == 503


def test_callback_signature_covers_exact_body():
    body, headers = signed_callback({'userId': 'test', 'jobId': JOB}, 'secret')
    expected = hmac.new(b'secret', headers['X-Generation-Timestamp'].encode() + b'.' + body, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(expected, headers['X-Generation-Signature'])


def test_training_uses_job_directory_and_returns_final_samples(tmp_path, monkeypatch):
    source = tmp_path / 'input'; source.mkdir()
    output = tmp_path / 'output'; output.mkdir()
    run = Mock()
    monkeypatch.setattr(predict.subprocess, 'run', run)
    samples = predict.run_training(str(source), 'person', steps=800, output_dir=str(output))
    args = run.call_args.args[0]
    assert f'--instance_data_dir={source}' in args
    assert not any('--concepts_list' in arg for arg in args)
    assert samples == str(output / '800' / 'samples')


def test_worker_rejects_missing_samples_without_callback(tmp_path, monkeypatch):
    celery = SimpleNamespace(task=lambda **kw: lambda fn: fn)
    monkeypatch.setitem(sys.modules, 'celery_app', SimpleNamespace(celery_app=celery))
    monkeypatch.setitem(sys.modules, 'pyuploadcare', SimpleNamespace(conf=SimpleNamespace()))
    sys.modules.pop('tasks', None)
    tasks = importlib.import_module('tasks')
    monkeypatch.setattr(tasks, 'callback_config', lambda: ('https://example.com/callback', 'secret'))
    monkeypatch.setattr(tasks.requests, 'get', lambda *a, **kw: SimpleNamespace(status_code=200, content=b'image'))
    monkeypatch.setattr(tasks, 'run_training', lambda **kw: str(tmp_path / 'missing'))
    callback = Mock(); monkeypatch.setattr(tasks.requests, 'Session', callback)
    with pytest.raises(RuntimeError, match='samples directory'):
        tasks.generate_images_task(None, {'userId': 'user', 'jobId': JOB, 'instanceImages': [IMAGE]})
    callback.assert_not_called()


def test_worker_uploads_final_samples_and_sends_signed_job_callback(tmp_path, monkeypatch):
    celery = SimpleNamespace(task=lambda **kw: lambda fn: fn)
    monkeypatch.setitem(sys.modules, 'celery_app', SimpleNamespace(celery_app=celery))
    upload = Mock(return_value=SimpleNamespace(uuid=IMAGE))
    monkeypatch.setitem(sys.modules, 'pyuploadcare', SimpleNamespace(conf=SimpleNamespace(), Uploadcare=lambda:SimpleNamespace(upload=upload)))
    sys.modules.pop('tasks', None)
    tasks = importlib.import_module('tasks')
    monkeypatch.setattr(tasks, 'callback_config', lambda: ('https://example.com/callback', 'secret'))
    monkeypatch.setattr(tasks.requests, 'get', lambda *a, **kw: SimpleNamespace(status_code=200, content=b'image'))
    def train(**kwargs):
        assert Path(kwargs['instance_data']).is_dir()
        assert (Path(kwargs['instance_data']) / '0.jpg').read_bytes() == b'image'
        samples = Path(kwargs['output_dir']) / str(kwargs['steps']) / 'samples'
        samples.mkdir(parents=True)
        (samples / '0.png').write_bytes(b'generated')
        return str(samples)
    monkeypatch.setattr(tasks, 'run_training', train)
    session = Mock()
    session.post.return_value.status_code = 200
    context = Mock(); context.__enter__ = Mock(return_value=session); context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(tasks.requests, 'Session', lambda: context)
    result = tasks.generate_images_task(None, {'userId':'user', 'jobId':JOB, 'instanceImages':[IMAGE]})
    assert result['generatedUuids'] == [IMAGE]
    upload.assert_called_once()
    args, kwargs = session.post.call_args
    assert args[0] == 'https://example.com/callback'
    assert json.loads(kwargs['data']) == {'userId':'user', 'jobId':JOB, 'generatedUuids':[IMAGE]}
    assert kwargs['allow_redirects'] is False
    headers = kwargs['headers']
    expected = hmac.new(b'secret', headers['X-Generation-Timestamp'].encode()+b'.'+kwargs['data'], hashlib.sha256).hexdigest()
    assert headers['X-Generation-Signature'] == expected
