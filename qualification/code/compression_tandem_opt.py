import numpy as np
import pandas as pd
import plotly.express as px
import beccs as phd

###PSO VARIABLES###
w = 1                # inertia term
c1 = 0.8             # cognitive term
c2 = 2               # social term
target_error = 1e-6
n_iterations = 10    # number of interations
n_particles = 30     # number of particles

parameters = [
    phd.temperature(750, 'k'),   #[0] Tcomb
    phd.temperature(170, 'k'),   #[1] Tflue
    phd.pressure(1.01315, 'pa'), #[2] Pcomb
    0.8,                         #[3] yO2
    30,                          #[4] m_rankine
    phd.pressure(1.01315, 'pa'), #[5] PlowRankyne
    phd.pressure(2.5, 'pa'),     #[6] PexaustRankyne  
    phd.pressure(60, 'pa'),      #[7] PhighRankine
    phd.pressure(150, 'pa'),     #[8] PhighBrayton
    phd.pressure(1.01315, 'pa'), #[9] PlowBrayton
    phd.temperature(25, 'k'),    #[10] TlowRankine
    phd.temperature(25, 'k'),    #[11] TlowBrayton
    20.83,                       #[12] mBagasse
    9.73,                        #[13] mStraw
    phd.temperature(70, 'k'),    #[14] Twater in K      
    phd.pressure(80, 'pa')       #[15] scCO2 pressure       
]

bagasse = phd.LignoCel(0.075, 0.5, 0.405, 0.39, 0.37, 0.24, 0.02, parameters[12])
straw = phd.LignoCel(0.05, 0.25, 0.68, 0.48, 0.35, 0.17, 0.02, parameters[13])
biogas = phd.BioGas(0.6, 0.4, m=0.2651747054131219)
y_o2in = 0.21   # oxygen concentration in the atmospheric air

combustion = phd.CombustionChamber(parameters[0], parameters[2])   # initiate the combustion chamber class
combustion.make_flue_gases(bagasse, straw, biogas, parameters[3])  # formation of the flue gases

list_yo2 = np.linspace(y_o2in, 1, 30, True)  # generates a list of 30 values of oxygen molar concentration
list_combustion = list(map(lambda y_o2: combustion.make_flue_gases(bagasse, straw, biogas, y_o2), list_yo2))
list_dwt = list(map(lambda comb: comb.dewatering()[0], list_combustion))
optmized_spaces = np.array(list(map(lambda yO2, dwt: [yO2, phd.pso_simulation_tandem(phd.initiate_tandem(5, n_particles, parameters[15], dwt), n_iterations)[0]], list_yo2, list_dwt)))

df = pd.DataFrame(optmized_spaces)
df['yN2'] = 1 - df[0]
df['work'] = df[1]

###PLOT###
fig = px.line(df, x='yN2', y='work', title='Compression power (5 compressors)')
fig.update_layout(xaxis_title="N2 molar fraction", yaxis_title="Power in MW")
fig.write_html('optimized_power_consumption_5comp')
fig.show()