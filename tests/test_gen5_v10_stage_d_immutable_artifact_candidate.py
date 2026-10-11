#!/usr/bin/env python3
"""CPU-only Stage D writer/reader, no repo writes and no model execution."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import gen5_v10_stage_d_immutable_artifact_candidate as d1
from scripts import gen5_v10_e0e3_v3_runtime_verifier_candidate as v3
import test_gen5_v10_e0e3_v3_runtime_verifier_candidate as v3_tests


class TestStageDImmutableArtifactCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._v3 = v3_tests.TestV3UntrustedCPU()
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def setUp(self):
        self.t = tempfile.TemporaryDirectory(prefix='gen5-v10-d1-test-')
        self.addCleanup(self.t.cleanup)
        self.parent = Path(self.t.name)

    def fixture(self):
        return self._v3.fixture()

    def failure(self, with_selector=False):
        selector = self.fixture()['selector'] if with_selector else None
        return d1.failure_envelope(stage='FROZEN' if selector else 'PRESELECT',
                                   reason_code='RAW_RECONSTRUCTION_AUTH',
                                   message='failed before complete export',
                                   selector=selector)

    def test_static_only_contract(self):
        self.assertEqual(d1.BASE,'869b92b1f3b17f97f4d844a20a7c1388f24cb10f')
        self.assertEqual(d1.V3_BASE,v3.BASE)
        self.assertNotIn('run-bounded', Path(d1.__file__).read_text(encoding='utf-8'))

    def test_valid_v3_readback_not_promoted(self):
        path=self.parent/'run-d1'
        obj=d1.write_v3_bundle(self.fixture(),path)
        verify=d1.verify_artifact(path)
        self.assertEqual(verify['kind'],'V3_UNAUTHENTICATED')
        self.assertEqual(obj['manifest_file_sha256'],verify['manifest_file_sha256'])
        self.assertEqual(verify['verdict']['LEGACY_FULL_RECORD'],'FAIL')
        self.assertEqual(verify['verdict']['V10_E0_PROVENANCE'],'NOT_EVALUATED')
        self.assertEqual(verify['verdict']['V10_E2_PAYLOAD'],'NOT_EVALUATED')
        self.assertEqual(len(list(path.iterdir())),6)

    def test_reject_existing_no_overwrite(self):
        path=self.parent/'run-d1'
        d1.write_v3_bundle(self.fixture(),path)
        before={p.name:p.read_bytes() for p in path.iterdir()}
        with self.assertRaisesRegex(d1.ArtifactBlocked,'COLLISION'):
            d1.write_v3_bundle(self.fixture(),path)
        self.assertEqual(before,{p.name:p.read_bytes() for p in path.iterdir()})

    def test_failure_only_with_preselect(self):
        path=self.parent/'failure-d1'
        d1.write_failure(self.failure(),path)
        r=d1.verify_artifact(path)
        self.assertEqual(r['kind'],'FAILURE_ONLY')
        self.assertEqual(r['verdict']['failure_reason'],'RAW_RECONSTRUCTION_AUTH')
        self.assertEqual(len(list(path.iterdir())),4)
        self.assertNotIn('LEGACY_FULL_RECORD',r['verdict'])

    def test_failure_only_with_selector(self):
        path=self.parent/'failure-d1'
        d1.write_failure(self.failure(with_selector=True),path)
        self.assertEqual(d1.verify_artifact(path)['kind'],'FAILURE_ONLY')

    def test_bogus_selector_blocks_before_write(self):
        obj=self.failure(with_selector=True)
        obj['selector']['target_cells'].reverse()
        with self.assertRaises(d1.ArtifactBlocked):
            d1.write_failure(obj,self.parent/'bad')
        self.assertFalse((self.parent/'bad').exists())

    def test_unsupported_failure_promotion_blocks(self):
        obj=self.failure()
        obj['V10_E0_PROVENANCE']='PASS'
        with self.assertRaisesRegex(d1.ArtifactBlocked,'FALSE_SUCCESS'):
            d1.write_failure(obj,self.parent/'bad')
        self.assertFalse((self.parent/'bad').exists())

    def test_forged_v3_blocks_before_create(self):
        obj=self.fixture()
        obj['engines']['v10']['records'][0]['R']=float('nan')
        with self.assertRaises(Exception):
            d1.write_v3_bundle(obj,self.parent/'bad')
        self.assertFalse((self.parent/'bad').exists())

    def test_wrong_report_tamper_detected(self):
        path=self.parent/'run-d1'
        d1.write_v3_bundle(self.fixture(),path)
        p=path/'verification.v3.json'
        obj=json.loads(p.read_text('utf-8'))
        obj['V10_E0_PROVENANCE']='PASS'
        p.write_bytes(d1._json_bytes(obj))
        with self.assertRaisesRegex(d1.ArtifactBlocked,'MANIFEST_FILE_TAMPER'):
            d1.verify_artifact(path)

    def test_tampered_sidecar_blocks(self):
        path=self.parent/'run-d1';d1.write_v3_bundle(self.fixture(),path)
        (path/'bundle.v3.json.sha256').write_text('0'*64+'  bundle.v3.json\n')
        with self.assertRaisesRegex(d1.ArtifactBlocked,'PHYSICAL_SHA256'):
            d1.verify_artifact(path)

    def test_missing_one_file_blocks(self):
        path=self.parent/'run-d1';d1.write_v3_bundle(self.fixture(),path)
        (path/'verification.v3.json.sha256').unlink()
        with self.assertRaisesRegex(d1.ArtifactBlocked,'EXTRA_OR_MISSING'):
            d1.verify_artifact(path)

    def test_extra_file_blocks(self):
        path=self.parent/'run-d1';d1.write_v3_bundle(self.fixture(),path)
        (path/'injected.txt').write_text('hello')
        with self.assertRaisesRegex(d1.ArtifactBlocked,'EXTRA_OR_MISSING'):
            d1.verify_artifact(path)

    def test_duplicate_json_keys_block(self):
        path=self.parent/'run-d1';d1.write_v3_bundle(self.fixture(),path)
        (path/'artifact_manifest.d1.json').write_bytes(b'{"a":1,"a":2}\n')
        with self.assertRaisesRegex(d1.ArtifactBlocked,'DUPLICATE_JSON_KEY'):
            d1.verify_artifact(path)

    def test_symlink_output_root_blocks(self):
        base=self.parent/'legit';base.mkdir()
        link=self.parent/'linked'
        try:
            link.symlink_to(base,target_is_directory=True)
        except (OSError,NotImplementedError):
            # Windows can deny creating a real symlink without privileges.
            # Exercise exactly the symlink-rejection branch, never skip it.
            link.mkdir()
            real_is_symlink = Path.is_symlink
            def simulated_is_symlink(path):
                return path == link or real_is_symlink(path)
            with patch.object(Path,'is_symlink',simulated_is_symlink):
                with self.assertRaisesRegex(d1.ArtifactBlocked,'PARENT_'):
                    d1.write_failure(self.failure(),link/'run')
        else:
            with self.assertRaisesRegex(d1.ArtifactBlocked,'PARENT_'):
                d1.write_failure(self.failure(),link/'run')
        self.assertFalse((link/'run').exists())

    def test_target_creation_race_cannot_replace_existing_directory(self):
        dest=self.parent/'race-run'
        original_publish = d1._publish_no_replace
        def inject_collision(staging, target):
            target.mkdir()  # Independent writer wins after pre-publish check.
            # Keep the destination empty: POSIX os.rename would overwrite it.
            return original_publish(staging, target)
        with patch.object(d1,'_publish_no_replace',inject_collision):
            with self.assertRaisesRegex(d1.ArtifactBlocked,'TARGET_COLLISION_AT_PUBLISH'):
                d1.write_failure(self.failure(),dest)
        self.assertTrue(dest.is_dir())
        self.assertEqual(list(dest.iterdir()), [])

    def test_no_inside_code_worktree(self):
        repo=Path(d1.__file__).resolve().parents[1]
        dest=repo/'stage-d-forbidden-root'
        with self.assertRaisesRegex(d1.ArtifactBlocked,'INSIDE_CODE_WORKTREE'):
            d1.write_failure(self.failure(),dest)
        self.assertFalse(dest.exists())

    def test_manual_repo_restriction(self):
        with self.assertRaisesRegex(d1.ArtifactBlocked,'INSIDE_CODE_WORKTREE'):
            d1.write_failure(self.failure(),self.parent/'inside',repo=self.parent)

    def test_wrong_failure_phase_blocks(self):
        obj=self.failure();obj['phase']='V10'
        with self.assertRaisesRegex(d1.ArtifactBlocked,'MISSING_SELECTOR'):
            d1.write_failure(obj,self.parent/'no')

    def test_runtime_self_assertions_remain_not_e0(self):
        obj=self._v3.pseudo_runtime_claim_fixture()
        path=self.parent/'runtime-claim'
        d1.write_v3_bundle(obj,path)
        result=d1.verify_artifact(path)['verdict']
        self.assertFalse(result['input_source_authenticated'])
        self.assertEqual(result['V10_E0_PROVENANCE'],'NOT_EVALUATED')

    def test_artifact_hash_cannot_prove_origin(self):
        path=self.parent/'run-d1'
        d1.write_v3_bundle(self.fixture(),path)
        report=d1.verify_artifact(path)
        self.assertEqual(report['status'],'OFFLINE_INTEGRITY_ONLY_NO_PROVENANCE_ATTESTATION')


if __name__ == '__main__':unittest.main()
