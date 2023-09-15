// SPDX-License-Identifier: LGPL-3.0-or-later
const deepmd = require('deepmd-kit');

deepmd.convert_pbtxt_to_pb(
    __dirname + '/../../tests/infer/deeppot.pbtxt', 'deeppot.pb');

const dp = new deepmd.DeepPot('deeppot.pb');

const coord = [1., 0., 0., 0., 0., 1.5, 1., 0., 3.];
const atype = [1, 0, 1];
const cell = [10., 0., 0., 0., 10., 0., 0., 0., 10.];

const v_coord = new deepmd.vectord(coord.length);
const v_atype = new deepmd.vectori(atype.length);
const v_cell = new deepmd.vectord(cell.length);
for (var i = 0; i < coord.length; i++) v_coord.set(i, coord[i]);
for (var i = 0; i < atype.length; i++) v_atype.set(i, atype[i]);
for (var i = 0; i < cell.length; i++) v_cell.set(i, cell[i]);

var energy = 0.0
var v_forces = new deepmd.vectord();
var v_virials = new deepmd.vectord();

energy = dp.compute(energy, v_forces, v_virials, v_coord, v_atype, v_cell);

console.log('energy:', energy);
console.log(
    'forces:', [...Array(v_forces.size()).keys()].map(i => v_forces.get(i)));
console.log(
    'virials:', [...Array(v_virials.size()).keys()].map(i => v_virials.get(i)));
