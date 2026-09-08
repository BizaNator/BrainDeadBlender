"""Shared reference resolution: stale artifacts cannot silently become defaults."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('reference_paths',
    Path(__file__).resolve().parents[1] / 'braindead_blender/reference_paths.py')
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


class ReferencePathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'reference.json'
        self.entry = {'profile': 'NATIVE_PLAYER'}
        for kind in ('fbx', 'contract'):
            path = self.root / ('player.' + kind)
            path.write_bytes(kind.encode())
            self.entry[kind] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.write()

    def write(self):
        self.manifest.write_text(json.dumps({'schema_version': '1.0.0', 'authoring_derivatives': [self.entry]}))

    def resolve(self):
        return resolver.reference_paths('NATIVE_PLAYER', manifest_path=self.manifest)

    def test_relative_artifacts_resolve(self):
        self.assertEqual(self.resolve(), (self.root / 'player.fbx', self.root / 'player.contract'))

    def test_changed_reference_bytes_are_rejected(self):
        (self.root / 'player.fbx').write_bytes(b'stale replacement')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.resolve()

    def test_changed_provenance_bytes_are_rejected(self):
        (self.root / 'player.contract').write_bytes(b'changed provenance')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.resolve()

    def test_missing_derivative_does_not_fall_back(self):
        self.entry['profile'] = 'NATIVE_DEVICE'
        self.write()
        with self.assertRaisesRegex(ValueError, 'Expected one'):
            self.resolve()

    def test_explicit_missing_manifest_is_rejected(self):
        self.manifest.unlink()
        with self.assertRaises(FileNotFoundError):
            self.resolve()

    def test_legacy_installation_without_manifest_keeps_device_default(self):
        with patch.dict('os.environ', {}, clear=True):
            fbx, contract = resolver.reference_paths(brains_root=self.root)
        self.assertEqual(fbx.name, 'CP_Device_Mannequin_named.fbx')
        self.assertEqual(contract.name, 'CP_Device_Mannequin_named.provenance.json')


if __name__ == '__main__':
    unittest.main()
