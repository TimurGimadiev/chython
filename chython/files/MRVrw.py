# -*- coding: utf-8 -*-
#
#  Copyright 2017-2024 Ramil Nugmanov <nougmanoff@protonmail.com>
#  Copyright 2025 Timur Gimadiev <timur.gimadiev@gmail.com>
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
from collections import defaultdict
from io import StringIO, BytesIO, TextIOWrapper, BufferedIOBase, BufferedReader
from itertools import count, islice, chain
from lxml.etree import iterparse, QName, tostring
from pathlib import Path
from typing import Union, List, Iterator, Dict, Optional
from ._convert import create_molecule, create_reaction
from ._mapping import postprocess_parsed_molecule, postprocess_parsed_reaction
from .mdl import postprocess_molecule
from ..containers import MoleculeContainer, ReactionContainer, MarkushContainer
from ..exceptions import EmptyMolecule, EmptyReaction


bond_map = {8: '1" queryType="Any', 4: 'A', 1: '1', 2: '2', 3: '3',
            'Any': 8, 'any': 8, 'A': 4, 'a': 4, '1': 1, '2': 2, '3': 3}


def xml_dict(parent_element, stop_list=None):
    stop_list = set() if stop_list is None else set(stop_list)
    out = {}
    for x, y in parent_element.items():
        y = y.strip()
        if y:
            x = '@%s' % x.strip()
            out[x] = y

    text = []
    if len(parent_element):
        elements_grouped = defaultdict(list)
        for element in parent_element:
            name = QName(element).localname
            if name in stop_list:
                text.append(tostring(element, encoding=str, with_tail=False))
            else:
                elements_grouped[name].append(element)

            if element.tail:
                t = element.tail.strip()
                if t:
                    text.append(t)

        for element_tag, element_group in elements_grouped.items():
            if len(element_group) == 1:
                out[element_tag] = xml_dict(element_group[0], stop_list)
            else:
                out[element_tag] = [xml_dict(x, stop_list) for x in element_group]

    if parent_element.text:
        t = parent_element.text.strip()
        if t:
            text.insert(0, t)
    if text:
        out['$'] = ''.join(text)

    return out


class MRVRead:
    """
    ChemAxon MRV files reader. works similar to opened file object. support `with` context manager.
    on initialization accept opened in binary mode file, string path to file,
    pathlib.Path object or another binary buffered reader object
    """
    molecule_cls = MoleculeContainer
    reaction_cls = ReactionContainer
    markush_cls = MarkushContainer

    def __init__(self, file, *, ignore: bool = True, remap: bool = False,
                 calc_cis_trans: bool = False, ignore_stereo: bool = False, ignore_bad_isotopes: bool = False):
        """
        :param ignore: Skip some checks of data or try to fix some errors.
        :param remap: Remap atom numbers started from one.
        :param calc_cis_trans: Calculate cis/trans marks from 2d coordinates.
        :param ignore_stereo: Ignore stereo data.
        :param ignore_bad_isotopes: reset invalid isotope mark to non-isotopic.
        """
        if isinstance(file, str):
            self.__file = open(file, 'rb')
            self.__is_buffer = False
        elif isinstance(file, Path):
            self.__file = file.open('rb')
            self.__is_buffer = False
        elif isinstance(file, (BytesIO, BufferedReader, BufferedIOBase)):
            self.__file = file
            self.__is_buffer = True
        else:
            raise TypeError('invalid file. BytesIO, BufferedReader and BufferedIOBase subclasses expected')
        self.__ignore = ignore
        self.__remap = remap
        self.__calc_cis_trans = calc_cis_trans
        self.__ignore_stereo = ignore_stereo
        self.__ignore_bad_isotopes = ignore_bad_isotopes
        self.__tell = 0
        self.__xml = iterparse(self.__file, tag='{*}MChemicalStruct')
        self.__buffer = None

    def read(self, amount: Optional[int] = None) -> List[Union[ReactionContainer, MoleculeContainer]]:
        """
        Parse whole file

        :param amount: number of records to read
        """
        if amount:
            return list(islice(iter(self), amount))
        return list(iter(self))

    def prepare_molecule(self, data, meta=None, rgroup=None):

        tmp = parse_molecule(data, rgroup)
        postprocess_parsed_molecule(tmp, remap=self.__remap, ignore=self.__ignore)
        parse_sgroup(data, tmp)
        mol = create_molecule(tmp, ignore_bad_isotopes=self.__ignore_bad_isotopes, _cls=self.molecule_cls)
        if not self.__ignore_stereo:
            postprocess_molecule(mol, tmp, calc_cis_trans=self.__calc_cis_trans)
        if meta:
            mol.meta.update(meta)
        return mol

    def read_structure(self, *, current: bool = True):
        """
        Read Reaction or Molecule container.

        :param current: return current structure if already parsed, otherwise read next
        """
        data = self._read_block(current=current)
        meta = self.read_metadata()
        log = []

        if 'molecule' in data and isinstance(data['molecule'], dict):
            datamol = data['molecule']
            mol = self.prepare_molecule(datamol, meta)
            print(mol)
            if 'Rgroup' not in data:
                return mol
            else:
                substituents = []
                data = data['Rgroup']
                for rgroup in data:
                    r_idx = int(rgroup['@rgroupID'])
                    if isinstance(rgroup['molecule'], dict):  # if single molecule, list will be omitted
                        molecules = [rgroup['molecule']]
                    else:
                        molecules = rgroup['molecule']
                    for moldata in molecules:
                        sub = self.prepare_molecule(moldata, rgroup=r_idx)
                        substituents.append(sub)
                return MarkushContainer.from_molecule(mol, substituents=substituents)


        elif 'reaction' in data and isinstance(data['reaction'], dict):
            data = data['reaction']
            tmp = {'reactants': [], 'products': [], 'reagents': [], 'log': log, 'title': data.get('@title')}

            n = 0
            for tag, group in (('reactantList', 'reactants'), ('productList', 'products'), ('agentList', 'reagents')):
                if tag in data and 'molecule' in data[tag]:
                    molecule = data[tag]['molecule']
                    if isinstance(molecule, dict):
                        molecule = (molecule,)
                    for m in molecule:
                        n += 1
                        try:
                            tmp[group].append(parse_molecule(m))
                        except ValueError as e:
                            if isinstance(e, EmptyMolecule):
                                log.append(f'ignored empty molecule {n}')
                            elif self.__ignore:
                                log.append(f'ignored molecule {n} with {e}')
                            else:
                                raise

            if not tmp['reactants'] and not tmp['products'] and not tmp['reagents']:
                raise EmptyReaction

            postprocess_parsed_reaction(tmp, remap=self.__remap, ignore=self.__ignore)
            rxn = create_reaction(tmp, ignore_bad_isotopes=self.__ignore_bad_isotopes, _m_cls=self.molecule_cls,
                                  _r_cls=self.reaction_cls)
            if not self.__ignore_stereo:
                for mol, tmp in zip(rxn.molecules(), chain(tmp['reactants'], tmp['reagents'], tmp['products'])):
                    postprocess_molecule(mol, tmp, calc_cis_trans=self.__calc_cis_trans)
            if meta:
                rxn.meta.update(meta)
            return rxn
        else:
            raise ValueError('reaction or molecule expected')

    def read_metadata(self, *, current: bool = True) -> Dict[str, str]:
        """
        Read metadata block
        """
        data = self._read_block(current=current)
        if 'molecule' in data and isinstance(data['molecule'], dict):
            data = data['molecule']
        elif 'reaction' in data and isinstance(data['reaction'], dict):
            data = data['reaction']
        else:
            raise ValueError('reaction or molecule expected')

        if 'propertyList' in data and 'property' in data['propertyList']:
            data = data['propertyList']['property']
            meta = {}
            if isinstance(data, dict):
                key = data['@title']
                val = data['scalar']['$'].strip()
                if key and val:
                    meta[key] = val
                else:
                    meta['chython_unparsed_metadata'] = [data]
            else:
                for x in data:
                    key = x['@title']
                    val = x['scalar']['$'].strip()
                    if key and val:
                        meta[key] = val
                    else:
                        if 'chython_unparsed_metadata' not in meta:
                            meta['chython_unparsed_metadata'] = []
                        meta['chython_unparsed_metadata'].append(x)
        else:
            return {}

    def close(self, force: bool = False):
        """
        Close opened file

        :param force: force closing of externally opened file or buffer
        """
        if not self.__is_buffer or force:
            self.__file.close()

    def tell(self):
        """
        Number of records processed from the original file
        """
        return self.__tell

    def __enter__(self):
        return self

    def __exit__(self, _type, value, traceback):
        self.close()

    def __iter__(self) -> Iterator[Union[ReactionContainer, MoleculeContainer]]:
        while True:
            try:
                yield self.read_structure(current=False)
            except ValueError:
                pass
            except EOFError:
                return

    def __next__(self) -> Union[ReactionContainer, MoleculeContainer]:
        return next(iter(self))

    def _read_block(self, *, current: bool = True) -> dict:
        if not current or not self.__buffer:
            self.__buffer = None
            try:
                e = next(self.__xml)[1]
            except StopIteration:
                raise EOFError
            self.__buffer = xml_dict(e)
            self.__tell += 1
            e.clear()
        return self.__buffer


def parse_molecule(data, rgroup: int = None):
    atoms, bonds, stereo = [], [], []
    log = []
    atom_map = {}
    if 'atom' in data['atomArray']:
        da = data['atomArray']['atom']
        if isinstance(da, dict):
            da = (da,)
        for n, atom in enumerate(da):
            atom_map[atom['@id']] = n
            element = atom['@elementType']
            isotope = int(atom['@isotope']) if '@isotope' in atom else None
            if element == "R" and atom['@rgroupRef']:
                isotope = int(atom['@rgroupRef'])
            elif element == "*":
                element = "R"
                if rgroup is not None:
                    isotope = rgroup
            atoms.append({'element': element,
                          'isotope': isotope,
                          'charge': int(atom.get('@formalCharge', 0)),
                          'is_radical': '@radical' in atom,
                          'parsed_mapping': int(atom.get('@mrvMap', 0))})
            if '@z3' in atom:
                atoms[-1].update(x=float(atom['@x3']), y=float(atom['@y3']), z=float(atom['@z3']))
            else:
                atoms[-1].update(x=float(atom['@x2']) / 2, y=float(atom['@y2']) / 2)
            if '@mrvQueryProps' in atom:
                raise ValueError('queries unsupported')
            if '@hydrogenCount' in atom:
                atoms[-1]['implicit_hydrogens'] = int(atom['@hydrogenCount'])
    else:
        atom = data['atomArray']
        for n, (_id, e) in enumerate(zip(atom['@atomID'].split(), atom['@elementType'].split())):
            atom_map[_id] = n
            atoms.append({'element': e})
        if '@z3' in atom:
            for a, x, y, z in zip(atoms, atom['@x3'].split(), atom['@y3'].split(), atom['@z3'].split()):
                a['x'] = float(x)
                a['y'] = float(y)
                a['z'] = float(z)
        else:
            for a, x, y in zip(atoms, atom['@x2'].split(), atom['@y2'].split()):
                a['x'] = float(x) / 2
                a['y'] = float(y) / 2
        if '@isotope' in atom:
            for a, x in zip(atoms, atom['@isotope'].split()):
                if x != '0':
                    a['isotope'] = int(x)
        if '@formalCharge' in atom:
            for a, x in zip(atoms, atom['@formalCharge'].split()):
                if x != '0':
                    a['charge'] = int(x)
        if '@mrvMap' in atom:
            for a, x in zip(atoms, atom['@mrvMap'].split()):
                if x != '0':
                    a['parsed_mapping'] = int(x)
        if '@radical' in atom:
            for a, x in zip(atoms, atom['@radical'].split()):
                if x != '0':
                    a['is_radical'] = True
        if '@mrvQueryProps' in atom:
            raise ValueError('queries unsupported')
    if not atoms:
        raise EmptyMolecule

    if 'bond' in data['bondArray']:
        db = data['bondArray']['bond']
        if isinstance(db, dict):
            db = (db,)
        for bond in db:
            order = bond_map[bond['@queryType' if '@queryType' in bond else '@order']]
            a1, a2 = bond['@atomRefs2'].split()
            if 'bondStereo' in bond:
                if '$' in bond['bondStereo']:
                    s = bond['bondStereo']['$']
                    if s == 'H':
                        stereo.append((atom_map[a1], atom_map[a2], -1))
                    elif s == 'W':
                        stereo.append((atom_map[a1], atom_map[a2], 1))
                    else:
                        log.append('invalid or unsupported stereo')
                else:
                    log.append('incorrect bondStereo tag')
            bonds.append((atom_map[a1], atom_map[a2], order))

    return {'atoms': atoms, 'bonds': bonds, 'stereo': stereo,
            'title': data.get('@title'), 'log': log, 'atom_map': atom_map}


def parse_sgroup(data, molecule):
    if 'molecule' in data:
        data = data['molecule']
        if isinstance(data, dict):
            data = (data,)

        sgroups = {}
        atom_map = molecule['mapping']
        atom_map = {k: atom_map[v] for k, v in molecule['atom_map'].items()}
        for x in data:
            if '@atomRefs' in x:
                atoms = [atom_map[x] for x in x['@atomRefs'].split()]
            elif 'AttachmentPointArray' in x:
                atoms = x['AttachmentPointArray']['attachmentPoint']
                if isinstance(atoms, dict):
                    atoms = (atoms,)
                atoms = [atom_map[x['@atom']] for x in atoms]
            else:
                continue
            tmp = {k[1:]: v for k, v in x.items() if k not in ('@atomRefs', '@id', '@molID') and k.startswith('@')}
            tmp['atoms'] = atoms
            sgroups[x['@id']] = tmp
        molecule['meta'] = sgroups

def parse_rgroup(data, molecule):
    """
    'Rgroup': [{'@rgroupID': '1',
                'molecule': [{'@molID': 'm2',
                              'atomArray': {'atom': [{'@id': 'a1',
                                                      '@elementType': 'C',
                                                      '@x2': '-3.562593333332587',
                                                      '@y2': '-1.1132399907607469'},
                                                     {'@id': 'a2',
                                                      '@elementType': 'C',
                                                      '@x2': '-4.896139989330879',
                                                      '@y2': '-0.3432399969207469'},
                                                     {'@id': 'a3',
                                                      '@elementType': 'C',
                                                      '@x2': '-4.896139989330879',
                                                      '@y2': '1.19657332409408'},
                                                     {'@id': 'a4',
                                                      '@elementType': 'C',
                                                      '@x2': '-2.2288600106691208',
                                                      '@y2': '1.19657332409408'},
                                                     {'@id': 'a5',
                                                      '@elementType': 'C',
                                                      '@x2': '-2.2288600106691208',
                                                      '@y2': '-0.3432399969207469'},
                                                     {'@id': 'a6',
                                                      '@elementType': '*',
                                                      '@x2': '-6.229835668979073',
                                                      '@y2': '-1.1132113172228375',
                                                      '@attachmentOrder': '1'}]},
                              'bondArray': {'bond': [{'@id': 'b1',
                                                      '@atomRefs2': 'a1 a2',
                                                      '@order': '1'},
                                                     {'@id': 'b2', '@atomRefs2': 'a1 a5', '@order': '1'},
                                                     {'@id': 'b3', '@atomRefs2': 'a2 a3', '@order': '1'},
                                                     {'@id': 'b4', '@atomRefs2': 'a3 a4', '@order': '1'},
                                                     {'@id': 'b5', '@atomRefs2': 'a4 a5', '@order': '1'},
                                                     {'@id': 'b6', '@atomRefs2': 'a2 a6', '@order': '1'}]}},
                             {'@molID': 'm3',
                              'atomArray': {'atom': [{'@id': 'a1',
                                                      '@elementType': 'C',
                                                      '@x2': '-4.14592666666592',
                                                      '@y2': '4.42842667590592'},
                                                     {'@id': 'a2',
                                                      '@elementType': 'C',
                                                      '@x2': '-5.479473322664212',
                                                      '@y2': '5.19842666974592'},
                                                     {'@id': 'a3',
                                                      '@elementType': 'C',
                                                      '@x2': '-5.479473322664212',
                                                      '@y2': '6.738239990760746'},
                                                     {'@id': 'a4',
                                                      '@elementType': 'C',
                                                      '@x2': '-2.812193344002454',
                                                      '@y2': '6.738239990760746'},
                                                     {'@id': 'a5',
                                                      '@elementType': 'C',
                                                      '@x2': '-2.812193344002454',
                                                      '@y2': '5.19842666974592'},
                                                     {'@id': 'a6',
                                                      '@elementType': '*',
                                                      '@x2': '-6.813169002312406',
                                                      '@y2': '4.428455349443829',
                                                      '@attachmentOrder': '1'}]},
                              'bondArray': {'bond': [{'@id': 'b1',
                                                      '@atomRefs2': 'a1 a2',
                                                      '@order': '1'},
                                                     {'@id': 'b2', '@atomRefs2': 'a1 a5', '@order': '1'},
                                                     {'@id': 'b3', '@atomRefs2': 'a2 a3', '@order': '1'},
                                                     {'@id': 'b4', '@atomRefs2': 'a3 a4', '@order': '1'},
                                                     {'@id': 'b5', '@atomRefs2': 'a4 a5', '@order': '1'},
                                                     {'@id': 'b6', '@atomRefs2': 'a2 a6', '@order': '1'}]}}]},
               {'@rgroupID': '2',
                'molecule': {'@molID': 'm4',
                             'atomArray': {'atom': [{'@id': 'a1',
                                                     '@elementType': 'N',
                                                     '@x2': '-11.020833333333334',
                                                     '@y2': '-5.988333333333333',
                                                     '@lonePair': '1'},
                                                    {'@id': 'a2',
                                                     '@elementType': 'C',
                                                     '@x2': '-12.354473333333333',
                                                     '@y2': '-5.218333333333334'},
                                                    {'@id': 'a3',
                                                     '@elementType': 'C',
                                                     '@x2': '-12.354473333333333',
                                                     '@y2': '-3.6783333333333332'},
                                                    {'@id': 'a4',
                                                     '@elementType': 'C',
                                                     '@x2': '-9.687193333333333',
                                                     '@y2': '-3.6783333333333332'},
                                                    {'@id': 'a5',
                                                     '@elementType': 'C',
                                                     '@x2': '-9.687193333333333',
                                                     '@y2': '-5.218333333333334'},
                                                    {'@id': 'a6',
                                                     '@elementType': '*',
                                                     '@x2': '-13.443417776360617',
                                                     '@y2': '-2.58938889030605',
                                                     '@attachmentOrder': '1'}]},
                             'bondArray': {'bond': [{'@id': 'b1', '@atomRefs2': 'a1 a2', '@order': '1'},
                                                    {'@id': 'b2', '@atomRefs2': 'a1 a5', '@order': '1'},
                                                    {'@id': 'b3', '@atomRefs2': 'a2 a3', '@order': '2'},
                                                    {'@id': 'b4', '@atomRefs2': 'a3 a4', '@order': '1'},
                                                    {'@id': 'b5', '@atomRefs2': 'a4 a5', '@order': '2'},
                                                    {'@id': 'b6', '@atomRefs2': 'a3 a6', '@order': '1'}]}}}]
    """



class MRVWrite:
    """
    ChemAxon MRV files writer. works similar to opened for writing file object. support `with` context manager.
    on initialization accept opened for writing in text mode file, string path to file,
    pathlib.Path object or another buffered writer object
    """
    def __init__(self, file, mapping: bool = True):
        """
        :param mapping: write atom mapping.
        """
        if isinstance(file, str):
            self.__file = open(file, 'w')
            self.__is_buffer = False
        elif isinstance(file, Path):
            self.__file = file.open('w')
            self.__is_buffer = False
        elif isinstance(file, (TextIOWrapper, StringIO)):
            self.__file = file
            self.__is_buffer = True
        else:
            raise TypeError('invalid file. '
                            'TextIOWrapper, StringIO, BytesIO, BufferedReader and BufferedIOBase subclasses possible')
        self.__writable = True
        self.__finalized = False
        self.__mapping = mapping

    def close(self, force=False):
        """
        Write close tag of MRV file and close opened file

        :param force: force closing of externally opened file or buffer
        """
        if not self.__finalized:
            self.__file.write('</cml>\n')
            self.__finalized = True
        if self.__writable:
            self.write = self.__write_closed
            self.__writable = False

        if not self.__is_buffer or force:
            self.__file.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, value, traceback):
        self.close()

    @staticmethod
    def __write_closed(_):
        raise ValueError('I/O operation on closed writer')

    def write(self, data: Union[ReactionContainer, MoleculeContainer]):
        """
        Write single molecule or reaction into file
        """
        self.__file.write('<cml>\n')
        self.__write(data)
        self.write = self.__write

    def __write(self, data):
        file = self.__file
        file.write('<MDocument><MChemicalStruct>')
        if isinstance(data, ReactionContainer):
            if not data._arrow:
                data.fix_positions()
            if data.name:
                file.write(f'<reaction title="{data.name}">')
            else:
                file.write('<reaction>')

            if data.meta:
                file.write('<propertyList>')
                for k, v in data.meta.items():
                    if isinstance(v, str):
                        v = f'<![CDATA[{v}]]>'
                    file.write(f'<property title="{k}"><scalar>{v}</scalar></property>')
                file.write('</propertyList>')

            c = count(1)
            for i, j in ((data.reactants, 'reactantList'), (data.reagents, 'agentList'),
                         (data.products, 'productList')):
                if not i:
                    continue
                file.write(f'<{j}>')
                for n, m in zip(c, i):
                    if m.name:
                        file.write(f'<molecule title="{m.name}" molID="m{n}">')
                    else:
                        file.write(f'<molecule molID="m{n}">')

                    self.__write_molecule(m)
                    file.write('</molecule>')
                file.write(f'</{j}>')

            file.write(f'<arrow type="DEFAULT" x1="{data._arrow[0] * 2:.4f}" y1="0" '
                       f'x2="{data._arrow[1] * 2:.4f}" y2="0"/></reaction>')
        elif not isinstance(data, MoleculeContainer):
            raise TypeError('MoleculeContainer expected')
        else:
            if data.name:
                file.write(f'<molecule title="{data.name}">')
            else:
                file.write('<molecule>')
            if data.meta:
                file.write('<propertyList>')
                for k, v in data.meta.items():
                    if isinstance(v, str):
                        v = f'<![CDATA[{v}]]>'
                    file.write(f'<property title="{k}"><scalar>{v}</scalar></property>')
                file.write('</propertyList>')

            self.__write_molecule(data)
            file.write('</molecule>')
        file.write('</MChemicalStruct></MDocument>\n')

    def __write_molecule(self, g):
        bg = g._bonds
        mapping = self.__mapping

        file = self.__file
        file.write('<atomArray>')
        for n, atom in g.atoms():
            x, y = atom.x, atom.y
            file.write(f'<atom id="a{n}" elementType="{atom.atomic_symbol}" x2="{x * 2:.4f}" y2="{y * 2:.4f}"')
            if mapping:
                file.write(f' mrvMap="{n}"')
            if atom.charge:
                file.write(f' formalCharge="{atom.charge}"')
            if atom.is_radical:
                file.write(' radical="monovalent"')
            if atom.isotope:
                file.write(f' isotope="{atom.isotope}"')
            if atom.implicit_hydrogens is not None:
                file.write(f' hydrogenCount="{atom.implicit_hydrogens}"')
            file.write('/>')
        file.write('</atomArray>')

        file.write('<bondArray>')
        wedge = defaultdict(set)
        n = 0  # empty wedge trick
        for n, (i, j, s) in enumerate(g._wedge_map, start=1):
            file.write(f'<bond id="b{n}" atomRefs2="a{i} a{j}" order="{bond_map[bg[i][j].order]}">'
                       f'<bondStereo>{s == 1 and "W" or "H"}</bondStereo></bond>')
            wedge[i].add(j)
            wedge[j].add(i)
        for i, j, bond in g.bonds():
            if j not in wedge[i]:
                n += 1
                file.write(f'<bond id="b{n}" atomRefs2="a{i} a{j}" order="{bond_map[bond.order]}"/>')
        file.write('</bondArray>')


__all__ = ['MRVRead', 'MRVWrite']
