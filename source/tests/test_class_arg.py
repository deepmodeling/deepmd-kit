import os,sys
import numpy as np
import unittest

from deepmd.common import ClassArg

class TestClassArg (unittest.TestCase) :
    def test_add (self) :
        ca = ClassArg().add('test', int)
        test_dict = {'test' :  10, 
                     'test1' : 20} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test':10})

    def test_add_multi (self) :
        ca = ClassArg()\
             .add('test',  int)\
             .add('test1', str)
        test_dict = {'test' :  10, 
                     'test1' : 'foo'} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test1':'foo', 'test':10})

    def test_add_multi_types (self) :
        ca = ClassArg()\
             .add('test',  [str, list])\
             .add('test1',  [str, list])
        test_dict = {'test' : [10,20], 'test1' : 10} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test':[10,20], 'test1':'10'})

    def test_add_type_cvt (self) :
        ca = ClassArg().add('test', float)
        test_dict = {'test' :  '10'} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test':10.0})

    def test_add_wrong_type_cvt (self) :
        ca = ClassArg().add('test', list)
        test_dict = {'test' :  10} 
        with self.assertRaises(TypeError):
            ca.parse(test_dict)

    def test_add_alias (self) :
        ca = ClassArg().add('test', str, alias = ['test1', 'test2'])
        test_dict = {'test2' :  'foo'} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test': 'foo'})

    def test_add_default (self) :
        ca = ClassArg().add('test', str, alias = ['test1', 'test2'], default = 'bar')
        test_dict = {'test3' :  'foo'} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test': 'bar'})

    def test_add_default_overwrite (self) :
        ca = ClassArg().add('test', str, alias = ['test1', 'test2'], default = 'bar')
        test_dict = {'test2' :  'foo'} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test': 'foo'})

    def test_add_must (self) :
        ca = ClassArg().add('test', str, must = True)
        test_dict = {'test2' :  'foo'} 
        with self.assertRaises(RuntimeError):
            ca.parse(test_dict)

    def test_add_none (self) :
        ca = ClassArg().add('test', int)
        test_dict = {'test2' :  'foo'} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test': None})

    def test_multi_add (self) :
        ca = ClassArg().add('test', int)
        test_dict = {'test2' :  'foo'} 
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test': None})
        ca.add('test2', str)
        ca.parse(test_dict)
        self.assertEqual(ca.get_dict(), {'test': None, 'test2':'foo'})




if __name__ == '__main__':
    unittest.main()
