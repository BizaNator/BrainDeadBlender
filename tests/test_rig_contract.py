"""Run with ordinary Python; no bpy import or external dependencies."""
import copy
import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'braindead_blender' / 'rig_contract.py'
spec = importlib.util.spec_from_file_location('rig_contract', MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    matrix = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    return {'bones': {'root': {'parent': None, 'matrix_world': matrix},
                       'index_01_l': {'parent': 'root', 'matrix_world': copy.deepcopy(matrix)}},
            'meshes': [{'name': 'hand', 'vertices': 3, 'polygons': 1,
                        'invalid_coordinates': 0, 'invalid_weights': 0,
                        'unweighted_vertices': 0, 'max_weight_sum_error': 0.,
                        'max_influences': 1, 'unknown_groups': [], 'bound': True,
                        'weight_totals': {'index_01_l': 3.}, 'uv_layers': ['UVMap'],
                        'materials': ['Skin'], 'invalid_morphs': [],
                        'morphs': {'BodyMass': {'max_delta_m': .02}}}]}


class RigContractTests(unittest.TestCase):
    def check(self, target, **kwargs):
        return audit.compare(fixture(), target, required_weighted=['index_01_l'], **kwargs)

    def test_valid_real_weights(self):
        self.assertTrue(self.check(fixture())['structural_pass'])

    def test_matching_names_do_not_hide_moved_rest_bone(self):
        target=fixture();target['bones']['index_01_l']['matrix_world'][0][3]=.001
        self.assertIn('rest_position:index_01_l', self.check(target)['errors'])

    def test_bone_roll_change_is_rejected(self):
        target=fixture();m=target['bones']['index_01_l']['matrix_world']
        m[0][0],m[0][1],m[1][0],m[1][1]=0.,-1.,1.,0.
        self.assertIn('rest_orientation:index_01_l', self.check(target)['errors'])

    def test_wrong_parent_is_rejected(self):
        target=fixture();target['bones']['index_01_l']['parent']=None
        self.assertIn('parent:index_01_l', self.check(target)['errors'])

    def test_empty_finger_group_cannot_pass(self):
        target=fixture();target['meshes'][0]['weight_totals']['index_01_l']=0.
        self.assertIn('missing_weight:index_01_l', self.check(target)['errors'])

    def test_unbound_empty_and_unweighted_meshes_fail(self):
        for field,value,error in [('bound',False,'unbound'),('vertices',0,'empty_geometry'),
                                  ('unweighted_vertices',1,'unweighted_vertices')]:
            target=fixture();target['meshes'][0][field]=value
            self.assertIn(error+':hand',self.check(target)['errors'])

    def test_nan_and_bad_normalization_fail_closed(self):
        target=fixture();target['bones']['root']['matrix_world'][0][0]=float('nan')
        target['meshes'][0]['max_weight_sum_error']=.1
        errors=self.check(target)['errors']
        self.assertIn('invalid_matrix:root',errors)
        self.assertIn('weight_normalization:hand',errors)

    def test_missing_morph_and_uv_fail(self):
        target=fixture();target['meshes'][0]['morphs']={};target['meshes'][0]['uv_layers']=[]
        errors=self.check(target,required_morphs=['BodyMass'])['errors']
        self.assertIn('missing_morph:BodyMass',errors)
        self.assertIn('missing_uv:hand',errors)

    def test_engine_compatibility_is_never_implied(self):
        self.assertFalse(self.check(fixture())['engine_animation_tested'])

    def test_missing_reference_and_no_meshes_fail(self):
        self.assertFalse(audit.compare({'bones':{}},fixture())['structural_pass'])
        target=fixture();target['meshes']=[]
        self.assertFalse(self.check(target)['structural_pass'])


if __name__=='__main__':
    unittest.main()
