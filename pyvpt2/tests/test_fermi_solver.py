import numpy as np
import pytest
from psi4 import compare_values

import pyvpt2
from pyvpt2.fermi_solver import Interaction, State, fermi_solver


def test_identify_fermi():

    omega = [1544.3, 1601.8, 2956.6]
    phi_ijk = np.asarray([
    [[158.8875601556632, -247.29101465678508, -166.74685943583202],
    [-247.29101465678508, 120.09522082111998, -142.50632482892297],
    [-166.74685943583202, -142.50632482892297, -281.86663885753416]],
    [[-247.29101465678508, 120.09522082111998, -143.13648130744926],
    [120.09522082111998, -426.7578692752896, 52.4164500795531],
    [-143.13648130744926, 52.4164500795531, 151.5566843963656]],
    [[-166.74685943583202, -148.31003819977596, -281.86663885753416],
    [-148.31003819977596, 52.4164500795531, 151.5566843963656],
    [-281.86663885753416, 151.5566843963656, 2406.274767904205]]])
    n_modes = 3
    v_ind = [0,1,2]
    kwargs = pyvpt2.process_options_keywords()
    fermi_list = pyvpt2.identify_fermi(omega, phi_ijk, n_modes, v_ind, **kwargs)
    assert fermi_list.pop() == (2, (0,0))
    assert len(fermi_list) == 0

def test_identify_fermi2():

    omega = [1335.4895, 1382.7834, 1679.7417, 2031.0176, 3157.7172, 3230.2677]
    phi_ijk = np.asarray([[[-9.638978874938006e-06, -4.91046798410187e-10, -63.86048561325305, 22.365741184842907, 326.1885869565015, 2.948992273951842e-10],
                           [-4.91046798410187e-10, 7.632333745112542e-06, -7.334118522144819e-11, 2.9298018820487156e-11, -1.2908784841133729e-11, 3.276894944810298e-05],
                           [-63.86048561325304, -7.334118522144819e-11, 2.227467003410018e-06, -3.0223119533983954e-07, -4.101476797787468e-06, -2.1315276876208665e-11],
                           [22.365741184842907, 2.929801882048715e-11, -3.0223119533983954e-07, -1.0804368070280477e-05, -9.911306289547784e-06, 4.219177236914758e-13],
                           [326.1885869565015, -1.2908784841133729e-11, -4.101476797787468e-06, -9.911306289547784e-06, -3.576764734323845e-05, -9.337393824752662e-12],
                           [2.948992273951842e-10, 3.276894944810298e-05, -2.1315276876208665e-11, 4.219177236914758e-13, -9.337393824752662e-12, -3.6648324486038935e-05]],
                           [[-4.910467984101869e-10, 7.632333745112542e-06, -7.334118522144819e-11, 2.9298018820487156e-11, -1.2908784841133729e-11, 3.2768949448102985e-05],
                            [7.632333745112542e-06, -6.034535557394472e-10, 113.40148640342282, 14.243903516834969, 238.8822068161842, -1.0583256209478652e-09],
                            [-7.334118522144819e-11, 113.40148640342284, -2.2290000330421799e-10, -1.1757535273704656e-09, -5.250221987952811e-11, 195.4582712550821],
                            [2.9298018820487156e-11, 14.243903516834969, -1.1757535273704656e-09, -1.662667551954454e-09, 5.771525725966952e-11, -134.6088331325876],
                            [-1.2908784841133729e-11, 238.8822068161842, -5.250221987952811e-11, 5.771525725966952e-11, 2.6102093898555483e-11, -9.36706866872757],
                            [3.276894944810298e-05, -1.0583256209478652e-09, 195.45827125508217, -134.6088331325876, -9.36706866872757, -5.650230305670037e-10]],
                            [[-63.86048561325305, -7.334118522144819e-11, 2.227467003410018e-06, -3.0223119533983954e-07, -4.101476797787468e-06, -2.1315276876208665e-11],
                             [-7.334118522144819e-11, 113.40148640342284, -2.2290000330421796e-10, -1.1757535273704656e-09, -5.2502219879528105e-11, 195.45827125508217],
                             [2.227467003410018e-06, -2.2290000330421796e-10, -58.70237290069749, 106.82886371319948, 85.84987333432582, 2.640896348256802e-10],
                             [-3.022311953398395e-07, -1.1757535273704654e-09, 106.82886371319948, 120.0504149789765, -49.73907545377756, -1.1056540951046945e-10],
                             [-4.101476797787468e-06, -5.250221987952811e-11, 85.84987333432582, -49.73907545377756, 2.649774143110787, -6.350485058090885e-11],
                             [-2.1315276876208665e-11, 195.4582712550821, 2.640896348256802e-10, -1.1056540951046945e-10, -6.350485058090884e-11, 134.67824939632092]],
                            [[22.365741184842907, 2.9298018820487156e-11, -3.0223119533983954e-07, -1.0804368070280477e-05, -9.911306289547785e-06, 4.2191772369147573e-13],
                             [2.9298018820487156e-11, 14.243903516834969, -1.1757535273704656e-09, -1.662667551954454e-09, 5.771525725966952e-11, -134.60883313258762],
                             [-3.0223119533983954e-07, -1.1757535273704658e-09, 106.82886371319948, 120.05041497897652, -49.73907545377756, -1.1056540951046945e-10],
                             [-1.0804368070280477e-05, -1.662667551954454e-09, 120.05041497897652, 570.2499483204989, 68.38145250304258, 7.505541817020066e-10],
                             [-9.911306289547785e-06, 5.771525725966952e-11, -49.73907545377756, 68.38145250304258, 1.6254526801927538, 2.2103088189990558e-10],
                             [4.2191772369147573e-13, -134.60883313258762, -1.1056540951046943e-10, 7.505541817020066e-10, 2.2103088189990558e-10, -92.87158665787145]],
                            [[326.1885869565015, -1.2908784841133729e-11, -4.101476797787468e-06, -9.911306289547784e-06, -3.576764734323845e-05, -9.337393824752662e-12],
                             [-1.2908784841133729e-11, 238.8822068161842, -5.250221987952811e-11, 5.7715257259669505e-11, 2.6102093898555487e-11, -9.36706866872757],
                             [-4.101476797787468e-06, -5.250221987952811e-11, 85.84987333432582, -49.73907545377756, 2.6497741431107875, -6.350485058090885e-11],
                             [-9.911306289547784e-06, 5.771525725966952e-11, -49.73907545377756, 68.38145250304258, 1.6254526801927536, 2.2103088189990558e-10],
                             [-3.576764734323845e-05, 2.6102093898555483e-11, 2.6497741431107875, 1.6254526801927536, -1361.3413130517508, 1.2356505676981148e-10],
                             [-9.337393824752662e-12, -9.367068668727569, -6.350485058090885e-11, 2.2103088189990558e-10, 1.2356505676981148e-10, -1439.83716533028]],
                            [[2.948992273951842e-10, 3.2768949448102985e-05, -2.1315276876208665e-11, 4.2191772369147573e-13, -9.337393824752662e-12, -3.6648324486038935e-05],
                             [3.2768949448102985e-05, -1.0583256209478654e-09, 195.4582712550821, -134.6088331325876, -9.367068668727569, -5.650230305670037e-10],
                             [-2.1315276876208665e-11, 195.45827125508217, 2.640896348256802e-10, -1.1056540951046943e-10, -6.350485058090885e-11, 134.67824939632092],
                             [4.2191772369147573e-13, -134.60883313258762, -1.1056540951046943e-10, 7.505541817020068e-10, 2.2103088189990558e-10, -92.87158665787145],
                             [-9.337393824752662e-12, -9.367068668727569, -6.350485058090885e-11, 2.2103088189990555e-10, 1.2356505676981148e-10, -1439.83716533028],
                             [-3.6648324486038935e-05, -5.650230305670037e-10, 134.67824939632092, -92.87158665787145, -1439.83716533028, 8.414084278764533e-10]]])

    n_modes = 6
    v_ind = [0, 1, 2, 3, 4, 5]
    kwargs = pyvpt2.process_options_keywords()

    fermi_list = pyvpt2.identify_fermi(omega, phi_ijk, n_modes, v_ind, **kwargs)
    assert (5, (1,2)) in fermi_list
    assert (5, (2,1)) in fermi_list
    assert len(fermi_list) == 2

def test_fermi_solver():

    ref = {(1,2): 3047.2, (0,): 3116.3, (3,6): 2711.5, (5,): 2851.5, (2,6): 3004.3}

    interaction0 = Interaction(left=State(state=(0,), nu=3095.5), right=State(state=(1,2),
                            nu=3068.0), phi=-89.7, ftype=2)
    interaction1 = Interaction(left=State(state=(5,), nu=2828.0), right=State(state=(2,6),
                            nu=2987.4), phi=-146.7, ftype=2)
    interaction2 = Interaction(left=State(state=(5,), nu=2828.0), right=State(state=(3,6),
                            nu=2751.9), phi=185.8, ftype=2)
    fermi_list = [interaction0, interaction1, interaction2]

    state_list = fermi_solver(fermi_list)

    for state in state_list.keys():
        assert compare_values(ref[state], state_list[state], 0.1)

def test_fermi_solver_empty():

        fermi_list = []
        state_list = fermi_solver(fermi_list)
        assert len(state_list) == 0
