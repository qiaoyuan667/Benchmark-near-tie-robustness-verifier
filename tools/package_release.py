#!/usr/bin/env python3
"""Build a deterministic anonymous ZIP from the verified public manifest only."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import zipfile

import audit_release as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deny-term', action='append', default=[], metavar='PRIVATE_TERM')
    args = parser.parse_args()
    root, destination = audit.ROOT, args.output.resolve()
    if destination == root or root in destination.parents:
        parser.error('Place the ZIP outside the public repository')
    if destination.exists():
        parser.error('Output exists; choose a new filename to preserve the existing archive')
    if any(not term.strip() for term in args.deny_term):
        parser.error('Private exclusions must not be empty')
    issues, hashes, _ = audit.collect_tree(root, args.deny_term, check_local_identity=True)
    issues += audit.verify_manifest(root / audit.MANIFEST, hashes)
    if issues:
        parser.error('Release verification failed; run audit_release.py --verify for details')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='anonymous-artifact-', suffix='.zip',
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(set(hashes) | audit.EXCLUDED):
                path = root / name
                if not path.is_file():
                    raise ValueError('A required provenance record is missing')
                entry = zipfile.ZipInfo('family-dif-audit/' + name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, path.read_bytes(), compresslevel=9)
        if audit.audit_zip(temporary, hashes, args.deny_term):
            raise ValueError('Archive differs from the audited public files')
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print('Verified anonymous ZIP created.')
    print('SHA-256:', hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
