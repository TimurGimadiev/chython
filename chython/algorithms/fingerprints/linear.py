# -*- coding: utf-8 -*-
#
#  Copyright 2021, 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
#  Copyright 2021 Aleksandr Sizov <murkyrussian@gmail.com>
#  Copyright 2023 Timur Gimadiev <timur.gimadiev@gmail.com>
#  This file is part of chython.
#
#  chython is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from collections import defaultdict, deque
from math import log2
from typing import TYPE_CHECKING, Deque, Dict, List, Set, Tuple, Union

from numpy import uint8, zeros

if TYPE_CHECKING:
    from chython import CGRContainer, MoleculeContainer


class LinearFingerprint:
    __slots__ = ()
    """
    Linear fragments fingerprints.
    Transform structures into fingerprints based on linear fragments descriptors.
    Also count of fragments takes into account by activating multiple bits, but less or equal to 
    `number_bit_pairs`.

    For example `CC` fragment found 4 times and `number_bit_pairs` is set to 3.
    In this case will be activated 3 bits: for count 1, for count 2 and for count 3.
    This gives intersection in bits with another structure with only 2 `CC` fragments.
    """

    def _atom2identifiers(self, atom):
        raise NotImplementedError

    def linear_fingerprint(self, min_radius: int = 1, max_radius: int = 4,
                           length: int = 1024, number_active_bits: int = 2,
                           number_bit_pairs: int = 4):
        """
        Transform structures into array of binary features.

        :param min_radius: minimal length of fragments
        :param max_radius: maximum length of fragments
        :param length: bit string's length. Should be power of 2
        :param number_active_bits: number of active bits for each hashed tuple
        :param number_bit_pairs: describe how much repeating fragments we can count in hashable
               fingerprint (if number of fragment in molecule greater or equal this number,
               we will be
               activate only this number of fragments

        :return: array(n_features)
        """
        bits = self.linear_bit_set(min_radius, max_radius, length, number_active_bits,
                                   number_bit_pairs)
        fingerprints = zeros(length, dtype=uint8)
        fingerprints[list(bits)] = 1
        return fingerprints

    def linear_bit_set(self, min_radius: int = 1, max_radius: int = 4,
                       length: int = 1024, number_active_bits: int = 2,
                       number_bit_pairs: int = 4) -> Set[int]:
        """
        Transform structure into set of indexes of True-valued features.

        :param min_radius: minimal length of fragments
        :param max_radius: maximum length of fragments
        :param length: bit string's length. Should be power of 2
        :param number_active_bits: number of active bits for each hashed tuple
        :param number_bit_pairs: describe how much repeating fragments we can count in hashable
               fingerprint (if number of fragment in molecule greater or equal this number,
               we will be activate only this number of fragments
        """
        mask = length - 1
        log = int(log2(length))

        hashes = self.linear_hash_set(min_radius, max_radius, number_bit_pairs)
        active_bits = set()
        for tpl in hashes:
            active_bits.add(tpl & mask)
            if number_active_bits == 2:
                active_bits.add((tpl >> log) & mask)
            elif number_active_bits > 2:
                for _ in range(1, number_active_bits):
                    tpl >>= log  # shift
                    active_bits.add(tpl & mask)
        return active_bits

    def linear_hash_set(self, min_radius: int = 1, max_radius: int = 4,
                        number_bit_pairs: int = 4) -> Set[int]:
        """
        Transform structure into set of integer hashes of fragments with count information.

        :param min_radius: minimal length of fragments
        :param max_radius: maximum length of fragments
        :param number_bit_pairs: describe how much repeating fragments we can count in hashable
               fingerprint (if number of fragment in molecule greater or equal this number,
               we will be activate only this number of fragments
        """
        return {hash((*tpl, cnt)) for tpl, count in
                self._fragments(min_radius, max_radius).items()
                for cnt in range(min(len(count), number_bit_pairs))}

    def _chains(self: Union['MoleculeContainer', 'CGRContainer'], min_radius: int = 1,
                max_radius: int = 4) -> Set[
                Tuple[int, ...]]:
        queue: Deque[Tuple[int, ...]]  # typing
        atoms = self._atoms
        bonds = self._bonds

        if min_radius == 1:
            arr = {(x,) for x in atoms}
            if max_radius == 1:  # special case
                return arr
            else:
                queue = deque(arr)
        else:
            arr = set()
            queue = deque((x,) for x in atoms)

        while queue:
            now = queue.popleft()
            var = [now + (x,) for x in bonds[now[-1]] if x not in now]
            if var:
                if len(var[0]) < max_radius:
                    queue.extend(var)
                if len(var[0]) >= min_radius:
                    for frag in var:
                        rev = frag[::-1]
                        arr.add(frag if frag > rev else rev)
        return arr

    @staticmethod
    def _fragment(atoms: dict[int, int], bonds: dict[int, dict[int, 'Bond']], fragment):
        var = [atoms[fragment[0]]]
        for x, y in zip(fragment, fragment[1:]):
            var.append(int(bonds[x][y]))
            var.append(atoms[y])
        var = tuple(var)
        rev_var = var[::-1]
        if var <= rev_var:
            var = rev_var
        return var

    def _fragments(self: Union['MoleculeContainer', 'CGRContainer'], min_radius: int = 1,
                   max_radius: int = 4) -> \
            Dict[Tuple[int, ...],
                 List[Tuple[int, ...]]]:
        atoms = {idx: hash(self._atom2identifiers(atom))
                 for idx, atom in self.atoms()}

        bonds = self._bonds
        out = defaultdict(list)

        for frag in self._chains(min_radius, max_radius):
            var = self._fragment(atoms, bonds, frag)
            out[var].append(frag)
        return dict(out)

    def linear_fragments_smiles(self: Union['MoleculeContainer', 'CGRContainer'],
                                min_radius: int = 1, max_radius: int = 4) -> \
            Dict[str, List[Tuple[int, ...]]]:

        atoms = {idx: hash(self._atom2identifiers(atom))
                 for idx, atom in self.atoms()}

        bonds = self._bonds
        out = defaultdict(list)
        smiles_map = {}

        for frag in self._chains(min_radius, max_radius):
            var = self._fragment(atoms, bonds, frag)
            out[var].append(frag)
            if var in out:
                smi_direct = [self._format_atom(frag[0], 0, stereo=False)]
                for x, y in zip(frag, frag[1:]):
                    smi_direct.append(self._format_bond(x, y, 0, stereo=False, aromatic=False))
                    smi_direct.append(self._format_atom(y, 0, stereo=False))
                smi_back = smi_direct[::-1]
                smi_direct = "".join(smi_direct).upper()
                smi_back = "".join(smi_back).upper()
                smi_frag = sorted([smi_direct, smi_back])[0]
                smiles_map[var] = smi_frag
        out = dict(out)
        return {smiles_map[k]: v for k, v in out.items()}


__all__ = ['LinearFingerprint']
