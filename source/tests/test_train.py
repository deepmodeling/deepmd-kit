import unittest
from unittest.mock import patch, MagicMock

import deepmd
from deepmd.entrypoints.train import parse_auto_sel, parse_auto_sel_ratio, wrap_up_4, update_one_sel, update_sel

class TestTrain (unittest.TestCase) :
    def test_train_parse_auto_sel (self) :
        self.assertTrue(parse_auto_sel("auto"))
        self.assertTrue(parse_auto_sel("auto:12"))
        self.assertTrue(parse_auto_sel("auto:12:13"))
        self.assertFalse(parse_auto_sel([1,2]))
        self.assertFalse(parse_auto_sel("abc:12:13"))


    def test_train_parse_auto_sel_ratio (self) :
        self.assertEqual(parse_auto_sel_ratio("auto"), 1.1)
        self.assertEqual(parse_auto_sel_ratio("auto:1.2"), 1.2)
        with self.assertRaises(RuntimeError):
            parse_auto_sel_ratio("auto:1.2:1.3")
        with self.assertRaises(RuntimeError):
            parse_auto_sel_ratio("abc")
        with self.assertRaises(RuntimeError):
            parse_auto_sel_ratio([1,2,3])


    @patch("deepmd.entrypoints.train.get_sel")
    def test_update_one_sel(self, sel_mock):
        sel_mock.return_value = [10,20]
        jdata = {}
        descriptor = {
            'rcut': 6,
            'sel': "auto"
        }
        descriptor = update_one_sel(jdata, descriptor)
        # self.assertEqual(descriptor['sel'], [11,22])
        self.assertEqual(descriptor['sel'], [12,24])
        descriptor = {
            'rcut': 6,
            'sel': "auto:1.5"
        }
        descriptor = update_one_sel(jdata, descriptor)
        # self.assertEqual(descriptor['sel'], [15,30])
        self.assertEqual(descriptor['sel'], [16,32])


    @patch("deepmd.entrypoints.train.get_sel")
    def test_update_sel_hybrid(self, sel_mock):
        sel_mock.return_value = [10,20]
        jdata = {
            'model' : {
                'descriptor': {
                    'type' : 'hybrid',
                    'list' : [
                        {
                            'rcut': 6,
                            'sel': "auto"                            
                        },
                        {
                            'rcut': 6,
                            'sel': "auto:1.5"
                        }
                    ]
                }
            }
        }
        expected_out = {
            'model' : {
                'descriptor': {
                    'type' : 'hybrid',
                    'list' : [
                        {
                            'rcut': 6,
                            'sel': [12,24] 
                        },
                        {
                            'rcut': 6,
                            'sel': [16,32]
                        }
                    ]
                }
            }
        }
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)


    @patch("deepmd.entrypoints.train.get_sel")
    def test_update_sel(self, sel_mock):
        sel_mock.return_value = [10,20]
        jdata = {
            'model' : {
                'descriptor': {
                    'type' : 'se_e2_a',
                    'rcut': 6,
                    'sel': "auto"
                }
            }
        }
        expected_out = {
            'model' : {
                'descriptor': {
                    'type' : 'se_e2_a',
                    'rcut': 6,
                    'sel': [12,24] 
                }
            }
        }
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    
    def test_wrap_up_4(self):
        self.assertEqual(wrap_up_4(12), 3 * 4)
        self.assertEqual(wrap_up_4(13), 4 * 4)
        self.assertEqual(wrap_up_4(14), 4 * 4)
        self.assertEqual(wrap_up_4(15), 4 * 4)
        self.assertEqual(wrap_up_4(16), 4 * 4)
        self.assertEqual(wrap_up_4(17), 5 * 4)
        
