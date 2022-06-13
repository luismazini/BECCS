import numpy as np
from CoolProp.CoolProp import PropsSI as cp
import pandas as pd
import beccs as phd

y_o2in = 0.21
raw_df = pd.read_json('/home/luismazini/Documents/phd/oxycombustion/total_power_consumption.json')
list_rawdf = np.array(raw_df)

parameters = [
    phd.temperature(600, 'k'),   #[0] Tcomb
    phd.temperature(170, 'k'),   #[1] Tflue
    phd.pressure(1.01315, 'pa'), #[2] Pcomb
    0.8,                         #[3] yO2
    40,                          #[4] m_rankine
    phd.pressure(1.01315, 'pa'), #[5] PlowRankyne
    phd.pressure(2.5, 'pa'),     #[6] PexaustRankyne  
    phd.pressure(60, 'pa'),      #[7] PhighRankine
    phd.pressure(150, 'pa'),     #[8] PhighBrayton
    phd.pressure(1.01315, 'pa'), #[9] PlowBrayton
    phd.temperature(25, 'k'),    #[10] TlowRankine
    phd.temperature(25, 'k'),    #[11] TlowBrayton
    38.89,                       #[12] mBagasse
    9.73,                        #[13] mStraw
    40                           #[14] mBrayton
]

bagasse = phd.LignoCel(0.075, 0.5, 0.405, 0.39, 0.37, 0.24, 0.02, parameters[12])
straw = phd.LignoCel(0.05, 0.25, 0.68, 0.48, 0.35, 0.17, 0.02, parameters[13])
biogas = phd.BioGas(0.6, 0.4, m=0.2651747054131219)

combustion = phd.CombustionChamber(parameters[0], parameters[2])  

def power_cycle(y_o2):
    combustion.make_flue_gases(bagasse, straw, biogas, y_o2)
    mass_exergy_water_in = parameters[4] * phd.prop_cp['water']['chem_x']
    mass_exergy_water_rec = (combustion.mrec - combustion.flue.mflue*combustion.x_h2o) * phd.prop_cp['water']['chem_x']
    pt1_brayton = phd.ThermoPoint('T', parameters[11], 'P', parameters[9], 'CO2')
    pt6_rankine = phd.ThermoPoint('T', parameters[10], 'P', parameters[5], 'H2O')
    pump_rankine = phd.PumpOrCompressor(pt6_rankine, parameters[7], pt6_rankine.fld, 1, parameters[4])

    compressor_brayton = phd.PumpOrCompressor(pt1_brayton, parameters[8], pt1_brayton.fld, 1, parameters[14])

    exchange1 = phd.DoubleHeater(combustion.flue, 1,compressor_brayton.pt_out, 0, combustion.flue.mflue, parameters[14])
    exchange1.ua_inf()

    turb_brayton = phd.Turbine(exchange1.pt_cold2, parameters[9], exchange1.pt_cold2.fld, massflow=parameters[14])

    recuperator = phd.DoubleHeater(turb_brayton.pt_out, 0, pump_rankine.pt_out, 0, parameters[14], pump_rankine.m)
    recuperator.ua_inf()

    condenser = phd.SingleHeater(recuperator.pt_hot2, pt1_brayton, parameters[14])
    q_ex_cond = condenser.exergy()
    exchange2 = phd.DoubleHeater(exchange1.pt_hot2, 1, recuperator.pt_cold2, 0, exchange1.m_hot, recuperator.m_cold)
    exchange2.get_pt(parameters[1], 'hot')

    turb_rankine = phd.Turbine(exchange2.pt_cold2, parameters[6], exchange2.pt_cold2.fld, massflow=parameters[4])
    mass_exergy_water_out = - parameters[4]*(phd.ph_exergy(turb_rankine.pt_out.h / 1e3, turb_rankine.pt_out.s / 1e3, cp('H', 'P', phd.p0, 'T', phd.t0, 'water') / 1e3, cp('S', 'P', phd.p0, 'T', phd.t0, 'water') / 1e3) + phd.prop_cp['water']['chem_x'])
    
    wliq = turb_rankine.work() + turb_brayton.work() + compressor_brayton.work() + pump_rankine.work()

    return [wliq / 1e3, mass_exergy_water_in / 1e3, mass_exergy_water_out /1e3, mass_exergy_water_rec / 1e3, q_ex_cond / 1e3]

optmized_spaces = np.array(list(map(lambda p: [
    p[0],                                  # [0] Recovery factor
    p[1],                                  # [1] Concentration factor
    p[2],                                  # [2] Fraction of the ideal separation power
    p[3],                                  # [3] Biogas exergy content
    p[4],                                  # [4] Bagasse exergy content
    p[5],                                  # [5] Sraw exergy content
    p[6],                                  # [6] Dewatering heat exergy content in MW
    p[7],                                  # [7] Separation power in MW
    p[8],                                  # [8] sCO2 exergy content
    p[9],                                  # [9] N2 B exit exergy content in MW
    p[10],                                 # [10] O2 B exit exergy content in MW
    p[11],                                 # [11] N2 non-condensable exergy content in MW
    p[12],                                 # [12] Compression power in MW
    p[13],                                 # [13] Compression heat exergy in MW
    power_cycle(y_o2in * p[1])             # [14] Power from cycles, exergy content of Rankine water inflow and outflow, dewatering water outflow, and heat in T-Brayton condenser in MW'
    ], list_rawdf)))

df = pd.DataFrame(optmized_spaces)
df = df.dropna()
df.rename(columns = {
    0: 'Recovery factor (nO2out/nO2in)', 
    1: 'Concentration factor (yO2out/yO2in)', 
    2: 'Ideal separation power %',
    3: 'Biogas exergy content (MW)',
    4: 'Bagasse exergy content (MW)',
    5: 'Straw exergy content (MW)',
    6: 'Dewatering heat exergy content (MW)',
    7: 'Separation power in MW',
    8: 'Supercritical carbon dioxide exergy content (MW)',
    9: 'N2 outflow (separation) exergy content (MW)',
    10: 'O2 outflow (separation) exergy content (MW)',
    11: 'N2 outflow (non-condensables) exergy content (MW)',
    12: 'Compression power in MW',
    13: 'Compression heat exergy in MW'}, inplace = True)

df[['Power from cycles in MW', 'Exergy content of Rankine water inflow in MW', 
'Exergy content of Rankine water outflow in MW', 'Exergy content of dewatering water outflow in MW', 
'Exergy content of Heat in T-Brayton condenser in MW']] = df[14].tolist()

df['Exergy destruction in MW'] = df['Biogas exergy content (MW)'] 
+ df['Bagasse exergy content (MW)'] 
+ df['Straw exergy content (MW)'] 
+ df['Supercritical carbon dioxide exergy content (MW)'] 
+ df['N2 outflow (separation) exergy content (MW)'] 
+ df['N2 outflow (non-condensables) exergy content (MW)'] 
+ df['O2 outflow (separation) exergy content (MW)'] 
+ df['Exergy content of Rankine water inflow in MW'] 
+ df['Exergy content of Rankine water outflow in MW'] 
+ df['Exergy content of dewatering water outflow in MW'] 
+ df['Dewatering heat exergy content (MW)'] 
+ df['Exergy content of Heat in T-Brayton condenser in MW'] 
+ df['Compression heat exergy in MW'] 
- (df['Exergy content of Heat in T-Brayton condenser in MW'] 
+ df['Separation power in MW'] 
+ df['Compression power in MW']
+ df['Power from cycles in MW'])

df['Exergetic efficiency'] = (df['Separation power in MW'] 
+ df['Compression power in MW'] 
+ df['Power from cycles in MW'] 
- df['Supercritical carbon dioxide exergy content (MW)']) / (df['Biogas exergy content (MW)'] 
+ df['Bagasse exergy content (MW)'] 
+ df['Straw exergy content (MW)']) 

df['Total separation power in MW'] = df['Separation power in MW'] + df['Compression power in MW']
df['Power surplus in MW'] = df['Total separation power in MW'] + df['Power from cycles in MW']

df.drop(14, axis=1, inplace=True)
df.to_json('CCS.json')
