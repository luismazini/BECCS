import CoolProp
from CoolProp.Plots import PropertyPlot
from CoolProp.Plots import SimpleCompressionCycle
pp = PropertyPlot('HEOS::R134a', 'PH', unit_system='EUR')
pp.calc_isolines(CoolProp.iQ, num=11)
cycle = SimpleCompressionCycle('HEOS::R134a', 'PH', unit_system='EUR')
T0 = 280
pp.state.update(CoolProp.QT_INPUTS,0.0,T0-10)
p0 = pp.state.keyed_output(CoolProp.iP)
T2 = 310
pp.state.update(CoolProp.QT_INPUTS,1.0,T2+15)
p2 = pp.state.keyed_output(CoolProp.iP)
pp.calc_isolines(CoolProp.iT, [T0-273.15,T2-273.15], num=2)
cycle.simple_solve(T0, p0, T2, p2, 0.7, SI=True)
cycle.steps = 50
sc = cycle.get_state_changes()
pp.draw_process(sc)
import matplotlib.pyplot as plt
plt.close(cycle.figure)
pp.show()