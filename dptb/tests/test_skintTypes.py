from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types, all_onsite_ene_types
from dptb.utils.index_mapping import Index_Mapings

# add test for all_onsite_intgrl_types
def test_onsite_intgrl_types():
    proj_atom_anglr_m = {'B':['2s'],'N':['2s','2p']}
    indm = Index_Mapings(proj_atom_anglr_m)
    onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                        indm.Onsite_Ind_Mapings(onsitemode='strain',atomtype=['N','B'])

    all_onsite_int_types_dict, reducted_onsite_int_types, sk_onsite_ind_dict = all_onsite_intgrl_types(onsite_strain_index_map)

    all_onsite_int_types_dict_check = {'N-N-2s-2s-0': 'N-N-2s-2s-0',
                                 'N-N-2s-2p-0': 'N-N-2s-2p-0',
                                 'N-N-2p-2s-0': 'N-N-2s-2p-0',
                                 'N-N-2p-2p-0': 'N-N-2p-2p-0',
                                 'N-N-2p-2p-1': 'N-N-2p-2p-1',
                                 'N-B-2s-2s-0': 'N-B-2s-2s-0',
                                 'N-B-2s-2p-0': 'N-B-2s-2p-0',
                                 'N-B-2p-2s-0': 'N-B-2s-2p-0',
                                 'N-B-2p-2p-0': 'N-B-2p-2p-0',
                                 'N-B-2p-2p-1': 'N-B-2p-2p-1',
                                 'B-N-2s-2s-0': 'B-N-2s-2s-0',
                                 'B-B-2s-2s-0': 'B-B-2s-2s-0'}
    
    reducted_onsite_int_types_check = ['N-N-2s-2s-0',
                                 'N-N-2s-2p-0',
                                 'N-N-2p-2p-0',
                                 'N-N-2p-2p-1',
                                 'N-B-2s-2s-0',
                                 'N-B-2s-2p-0',
                                 'N-B-2p-2p-0',
                                 'N-B-2p-2p-1',
                                 'B-N-2s-2s-0',
                                 'B-B-2s-2s-0']
    
    sk_onsite_ind_dict_check = {'N-N': ['N-N-2s-2s-0', 'N-N-2s-2p-0', 'N-N-2p-2p-0', 'N-N-2p-2p-1'],
                          'N-B': ['N-B-2s-2s-0', 'N-B-2s-2p-0', 'N-B-2p-2p-0', 'N-B-2p-2p-1'],
                          'B-N': ['B-N-2s-2s-0'],
                          'B-B': ['B-B-2s-2s-0']}
    
    assert isinstance(all_onsite_int_types_dict, dict)
    assert isinstance(reducted_onsite_int_types, list)
    assert isinstance(sk_onsite_ind_dict, dict)
    
    assert all_onsite_int_types_dict == all_onsite_int_types_dict_check
    assert reducted_onsite_int_types == reducted_onsite_int_types_check
    assert sk_onsite_ind_dict == sk_onsite_ind_dict_check

    uniq_sktype = set(all_onsite_int_types_dict.values())
    assert len(uniq_sktype) == len(reducted_onsite_int_types)
    for ia in uniq_sktype:
        assert ia in reducted_onsite_int_types
    
    assert list(sk_onsite_ind_dict.keys()) == (['N-N', 'N-B', 'B-N', 'B-B'])
    assert onsite_strain_index_map.keys() == sk_onsite_ind_dict.keys()
    
    for ibt in sk_onsite_ind_dict.keys():
        for isk in  onsite_strain_index_map[ibt].keys():
            index = onsite_strain_index_map[ibt][isk]
            for ii in range(len(index)):
                skbondname = f'{ibt}-{isk}-{ii}'
                assert sk_onsite_ind_dict[ibt][index[ii]] == all_onsite_int_types_dict[skbondname]


# add test for all_onsite_ene_types


def test_onsite_ene_types():
    proj_atom_anglr_m = {'B':['2s'],'N':['2s','2p']}
    indm = Index_Mapings(proj_atom_anglr_m)
    onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                        indm.Onsite_Ind_Mapings(onsitemode='strain',atomtype=['N','B'])
    all_onsiteE_types_dict, reduced_onsiteE_types, onsiteE_ind_dict = all_onsite_ene_types(onsite_index_map)
    all_onsiteE_types_dict_check = {'N-2s-0': 'N-2s-0', 'N-2p-0': 'N-2p-0', 'B-2s-0': 'B-2s-0'}
    reduced_onsiteE_types_check = ['N-2s-0', 'N-2p-0', 'B-2s-0']
    onsiteE_ind_dict_check = {'N': ['N-2s-0', 'N-2p-0'], 'B': ['B-2s-0']}
    
    assert isinstance(all_onsiteE_types_dict, dict)
    assert isinstance(reduced_onsiteE_types, list)
    assert isinstance(onsiteE_ind_dict, dict)
    
    assert all_onsiteE_types_dict == all_onsiteE_types_dict_check
    assert reduced_onsiteE_types == reduced_onsiteE_types_check
    assert onsiteE_ind_dict == onsiteE_ind_dict_check
    
def test_skintTypes():
    envtype = ['N','B']
    bondtype = ['N','B']
    proj_atom_anglr_m={'N':['2s','2p'],'B':['2p']}
    indmap = Index_Mapings(proj_atom_anglr_m=proj_atom_anglr_m)
    bond_map, bond_num = indmap.Bond_Ind_Mapings()
    all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(bond_map)


    all_skint_types_check= {'N-N-2s-2s-0': 'N-N-2s-2s-0',
                            'N-N-2s-2p-0': 'N-N-2s-2p-0',
                            'N-N-2p-2s-0': 'N-N-2s-2p-0',
                            'N-N-2p-2p-0': 'N-N-2p-2p-0',
                            'N-N-2p-2p-1': 'N-N-2p-2p-1',
                            'N-B-2s-2p-0': 'N-B-2s-2p-0',
                            'N-B-2p-2p-0': 'N-B-2p-2p-0',
                            'N-B-2p-2p-1': 'N-B-2p-2p-1',
                            'B-N-2p-2s-0': 'N-B-2s-2p-0',
                            'B-N-2p-2p-0': 'N-B-2p-2p-0',
                            'B-N-2p-2p-1': 'N-B-2p-2p-1',
                            'B-B-2p-2p-0': 'B-B-2p-2p-0',
                            'B-B-2p-2p-1': 'B-B-2p-2p-1'}

    reducted_skint_types_check = ['N-N-2s-2s-0',
                                  'N-N-2s-2p-0',
                                  'N-N-2p-2p-0',
                                  'N-N-2p-2p-1',
                                  'N-B-2s-2p-0',
                                  'N-B-2p-2p-0',
                                  'N-B-2p-2p-1',
                                  'B-B-2p-2p-0',
                                  'B-B-2p-2p-1']

    sk_bond_ind_check = {'N-N': ['N-N-2s-2s-0', 'N-N-2s-2p-0', 'N-N-2p-2p-0', 'N-N-2p-2p-1'],
                         'N-B': ['N-B-2s-2p-0', 'N-B-2p-2p-0', 'N-B-2p-2p-1'],
                         'B-N': ['N-B-2s-2p-0', 'N-B-2p-2p-0', 'N-B-2p-2p-1'],
                         'B-B': ['B-B-2p-2p-0', 'B-B-2p-2p-1']}

    assert isinstance(all_skint_types_dict, dict)
    assert isinstance(reducted_skint_types, list)
    assert isinstance(sk_bond_ind_dict, dict)

    assert all_skint_types_dict == all_skint_types_check
    assert reducted_skint_types == reducted_skint_types_check
    assert sk_bond_ind_dict == sk_bond_ind_check
    

    uniq_sktype = set(all_skint_types_dict.values())
    assert len(uniq_sktype) == len(reducted_skint_types)
    for ia in uniq_sktype:
        assert ia in reducted_skint_types
    
    assert list(sk_bond_ind_dict.keys()) == (['N-N', 'N-B', 'B-N', 'B-B'])
    assert bond_map.keys() == sk_bond_ind_dict.keys()

    for ibt in sk_bond_ind_dict.keys():
        for isk in  bond_map[ibt].keys():
            index = bond_map[ibt][isk]
            for ii in range(len(index)):
                skbondname = f'{ibt}-{isk}-{ii}'
                assert sk_bond_ind_dict[ibt][index[ii]] == all_skint_types_dict[skbondname]

def test_onsiteint_types():
    envtype = ['N','B']
    bondtype = ['N','B']
    proj_atom_anglr_m = {'B':['2s'],'N':['2s','2p']}
    indmap = Index_Mapings(proj_atom_anglr_m=proj_atom_anglr_m)
    onsite_intgrl_index_map,_ = indmap._OnsiteStrain_Ind_Mapings(atomtypes=['N','B'])
    all_onsiteint_types_dict, reducted_onsiteint_types, sk_onsite_ind_dict = all_onsite_intgrl_types(onsite_intgrl_index_map)

    all_onsiteint_types_dict_check= {'N-N-2s-2s-0': 'N-N-2s-2s-0',
                            'N-N-2s-2p-0': 'N-N-2s-2p-0',
                            'N-N-2p-2s-0': 'N-N-2s-2p-0',
                            'N-N-2p-2p-0': 'N-N-2p-2p-0',
                            'N-N-2p-2p-1': 'N-N-2p-2p-1',
                            'N-B-2s-2s-0': 'N-B-2s-2s-0',
                            'N-B-2s-2p-0': 'N-B-2s-2p-0',
                            'N-B-2p-2s-0': 'N-B-2s-2p-0',
                            'N-B-2p-2p-0': 'N-B-2p-2p-0',
                            'N-B-2p-2p-1': 'N-B-2p-2p-1',
                            'B-N-2s-2s-0': 'B-N-2s-2s-0',
                            'B-B-2s-2s-0': 'B-B-2s-2s-0'}

    reducted_onsiteint_types_check = ['N-N-2s-2s-0',
                                      'N-N-2s-2p-0',
                                      'N-N-2p-2p-0',
                                      'N-N-2p-2p-1',
                                      'N-B-2s-2s-0',
                                      'N-B-2s-2p-0',
                                      'N-B-2p-2p-0',
                                      'N-B-2p-2p-1',
                                      'B-N-2s-2s-0',
                                      'B-B-2s-2s-0']

    sk_onsite_ind_dict_check = {'N-N': ['N-N-2s-2s-0', 'N-N-2s-2p-0', 'N-N-2p-2p-0', 'N-N-2p-2p-1'],
                                'N-B': ['N-B-2s-2s-0', 'N-B-2s-2p-0', 'N-B-2p-2p-0', 'N-B-2p-2p-1'],
                                'B-N': ['B-N-2s-2s-0'],
                                'B-B': ['B-B-2s-2s-0']}

    assert isinstance(all_onsiteint_types_dict, dict)
    assert isinstance(reducted_onsiteint_types, list)
    assert isinstance(sk_onsite_ind_dict, dict)

    assert all_onsiteint_types_dict == all_onsiteint_types_dict_check
    assert reducted_onsiteint_types == reducted_onsiteint_types_check
    assert sk_onsite_ind_dict == sk_onsite_ind_dict_check
    

    uniq_sktype = set(all_onsiteint_types_dict.values())
    assert len(uniq_sktype) == len(reducted_onsiteint_types_check)
    for ia in uniq_sktype:
        assert ia in reducted_onsiteint_types_check
    
    assert list(sk_onsite_ind_dict.keys()) == (['N-N', 'N-B', 'B-N', 'B-B'])
    assert onsite_intgrl_index_map.keys() == sk_onsite_ind_dict.keys()

    for ibt in sk_onsite_ind_dict.keys():
        for isk in  onsite_intgrl_index_map[ibt].keys():
            index = onsite_intgrl_index_map[ibt][isk]
            for ii in range(len(index)):
                skbondname = f'{ibt}-{isk}-{ii}'
                assert sk_onsite_ind_dict[ibt][index[ii]] == all_onsiteint_types_dict[skbondname]