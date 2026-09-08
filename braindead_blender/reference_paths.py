"""Resolve authoring artifacts from the shared skeleton reference manifest.

This module has no Blender dependency so CPU wrappers use the same resolver.
It contains paths for older studio installations, never skeletal transforms.
"""
import hashlib
import json
import os
from pathlib import Path
import re


PROFILES = ('NATIVE_DEVICE', 'NATIVE_PLAYER', 'FAB_UEFN')


def reference_paths(profile='NATIVE_DEVICE', *, manifest_path=None, brains_root=None):
    if profile not in PROFILES:
        raise ValueError('Unknown target profile: ' + profile)
    brains = Path(brains_root) if brains_root else (
        Path(r'B:\Brains') if os.name == 'nt' else Path('/mnt/tank/Studio/Brains'))
    explicit = manifest_path or os.environ.get('BDB_SKELETON_REFERENCE')
    manifest = Path(explicit) if explicit else brains / 'Characters/_uefn_reference/skeleton_reference_v1.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('schema_version') != '1.0.0':
            raise ValueError('Unsupported skeleton reference schema: ' + str(manifest))
        entries = [row for row in data.get('authoring_derivatives', []) if row.get('profile') == profile]
        if len(entries) != 1:
            raise ValueError('Expected one authoring derivative for ' + profile)
        entry = entries[0]

        def artifact(kind, required=True):
            record = entry.get(kind)
            if record is None and not required:
                return None
            if not isinstance(record, dict) or not isinstance(record.get('path'), str):
                raise ValueError('Missing reference artifact: ' + kind)
            digest = record.get('sha256', '')
            if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', digest):
                raise ValueError('Invalid reference artifact hash: ' + kind)
            path = (manifest.parent / record['path']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest.lower():
                raise ValueError('Reference artifact hash mismatch: ' + str(path))
            return path

        return artifact('fbx'), artifact('contract', required=profile != 'FAB_UEFN')
    if explicit:
        raise FileNotFoundError('Skeleton reference manifest not found: ' + str(manifest))
    # Existing installations keep working before the consolidated manifest lands.
    if profile == 'FAB_UEFN':
        return brains / 'Skills/char-designer/skm_uefn_mannequin.FBX', None
    if profile == 'NATIVE_PLAYER':
        folder = brains / 'Characters/_uefn_reference/native_contracts/full_20260908/authoring_player_v03'
        return folder / 'Fortnite_Player_authoring_v03.fbx', folder / 'reference_contract.json'
    folder = brains / 'Characters/_uefn_reference/native_contracts/codex_device_v01'
    return folder / 'CP_Device_Mannequin_named.fbx', folder / 'CP_Device_Mannequin_named.provenance.json'
