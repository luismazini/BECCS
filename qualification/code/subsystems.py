from xml.etree.ElementTree import PI
import numpy as np
from CoolProp.CoolProp import PropsSI as cp
from CoolProp.HumidAirProp import HAPropsSI as air_cp
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import fsolve, newton_krylov

#convert ton/h into kg/s and vice-versa
def mass_flow(m, s):
    if s.lower() == 'kg':
        m = m * 1000 / 3600
    elif s.lower() == 'ton':
        m = m * 3.6
    return m
#convert celsius to kelvin and vice-versa
def temperature(t, s):
    if s.lower() == 'c':
        t += -273.15
    elif s.lower() == 'k':
        t += 273.15
    return t
#convert bar to pascal or vice versa
def pressure(p, s):
    if s.lower() == 'bar':
        p = p / 100000
    elif s.lower() == 'pa':
        p = p * 100000
    return p
#function that calculates the exergy destruction in kW
def exergy_destruction(q_ex, w_ex, m_ex):
    q_sum, w_sum, m_sum = 0, 0, 0
    if q_ex!=0:
        #loop that calculates the sum of heat exergy contents
        q_sum = sum(q_ex)
    if w_ex!=0:
        #loop that calculates the sum of work exergy contents   
            w_sum = sum(w_ex)
    if m_ex!=0:
        #loop that calculates the sum of mass flow exergy contents   
            m_sum = sum(m_ex)
    return q_sum - w_sum + m_sum
#function that calculates the physical exergy
def ph_exergy(h, s, h0, s0):
    #h = enthalpy in J/kg
    #s = entropy in J/kg*K
    #0 index -> dead state
    return h - h0 - t0 * (s - s0) #in kJ/kg
#function that calculates the sum of the standart chemical exergy in kJ/kg
#source: Szargut, J., Morris, D.R., Steward, F.R., 1987. Exergy Analysis of Thermal, Chemical,
#and Metallurgical Processes. Hemisphere Publishing, New York.
def ch_solid_exergy(lhv_fuel, beta, bch_s, ls, x_s, bch_a, x_a, bch_w, x_w):
    #hhv is the high heat value of the fuel expressed in kJ/kg
    #beta is the beta index of the chemical exergy of solid fuels
    #bch_s is the sulfur chemical exergy content expressed in kJ/kg
    #ls is the latent heat value of sulfur expressed in kJ/kg
    #x_s is the nitrogen mass fraction expressed in decimals
    #bch_a is the ash chemical exergy content expressed in kJ/kg
    #x_a is the ash mass fraction expressed in decimals
    #bch_w is the water chemical exergy content expressed in kJ/kg
    #x_w is the moisture mass fraction expressed in decimals
    return (lhv_fuel-lw)*beta+(bch_s-ls)*x_s+bch_a*x_a+bch_w*x_w
#function that calculates the HHV content of fuels in MJ/kg
#source: S. A. Channiwala and P. P. Parikh, “A unified correlation for estimating HHV of solid, 
#liquid and gaseous fuels,” Fuel, vol. 81, no. 8, pp. 1051–1063, May 2002, doi: 10.1016/S0016-2361(01)00131-4.
def hhv(c, h, s, o, n, a):
    #c is the carbon content expressed in %
    #h is the hydrogen content expressed in %
    #s is the sulfur content expressed in %
    #o is the oxygen content expressed in %
    #n is the nitrogen content expressed in %
    #a is the ash content expressed in %
    return 0.3491*c+1.1783*h+0.1005*s-0.1034*o-0.0151*n-0.0211*a
#function that calculates the LHV content of fuels in kJ/kg
#source: W. A. Bizzo and C. G. Sánchez, “Stoichiometry of combustion and gasification reactions 
#(Estequiometria das reações de combustão e gaseificação in Portuguese),” Biomass Gasification Technology (Tecnologia Da Gaseificação de Biomassa in Portuguese), p. 430, 2010.
def lhv(hhv_fuel, z_h, z_w):
    #hhv is the high heat value of the fuel expressed in kJ/kg
    #z_h is the hydrogen content expressed in decimals
    #z_w is the moisture content expressed in decimals
    return (hhv_fuel - lw * (9 * z_h - z_w)) * (1 - z_w)
#function that calculates beta index of the chemical exergy of solid fuels
#source: Szargut, J., Morris, D.R., Steward, F.R., 1987. Exergy Analysis of Thermal, Chemical,
#and Metallurgical Processes. Hemisphere Publishing, New York.
def beta_fuel(z_h, z_c, z_o, z_n):
    #z_h is the hydrogen content expressed in decimals
    #z_c is the carbon content expressed in decimals
    #z_o is the oxygen content expressed in decimals
    #z_n is the nitrogen content expressed in decimals
    return (1.0412 + 0.216 * z_h / z_c - 0.2499 * z_o / z_c * (1 + 0.7884 * z_h / z_c) + 0.045 * z_n / z_c) / (1 - 0.3035 * z_o / z_c)
#function that calculates the exergy content of mixtures in kJ/kg
def mix_exergy(a1, a2, M_mix, y1, y2, T):
    #a1 is the activity of substance 1
    #a2 is the activity of substance 2
    #M_mix is the molar mass of the mixture
    #y1 is the molar fraction of substance 1
    #y2 is the molar fraction of substance 2
    return R * T / M_mix * (y1 * np.log(a1) + y2 * np.log(a2))
def mass_to_molar_frac(x, M_s, M_w):
    #x is the solid mass fraction
    #M_s is the solid molar mass
    #M_w is the solvent molar mass
    return x / (x + (1 - x) * M_s / M_w)

def ph_exergy(h, s, h0, s0):
    #h = enthalpy in J/kg
    #s = entropy in J/kg*K
    #0 index -> dead state
    return h - h0 - t0 * (s - s0) #in kJ/kg

#function that calculates the ideal separation work
def ideal_sep_wk(subs, d_sub):
    #subs is the substance that you want to separate 100% percent
    w= d_sub[subs] * np.log(d_sub[subs])
    for cmp in d_sub:
        if cmp!=subs: 
            w = w + d_sub[cmp]  / (1 - d_sub[subs]) * np.log((1 - d_sub[subs]))
    return t0 * R / 1000 * w / cp('M', 'T', t0, 'P', p0, 'air') #in kJ/kg
#correlation that calculates the specific heat of the sucrose-water mixture at t temperature (in celsius degrees)
#x = concentration of solid mass
#source: 
def cp_kadlec(x, t):
    #x is the solid mass fraction
    #t is the temperature in ºC
    return  2.326 * (x / 10 * (100 + x) / (900 - 8 * x) + 1.8 * t * (1 - x / 10 * (0.6 - 0.0009 * t))) #in kJ/kg
#correlation that calculates the enthalpy of the sucrose-water mixture at t temperature (in celsius degrees)
#x=concentration of solid mass
#source: S. Peacock, “Predicting physical properties of factory juices and syrups,” 
#International Sugar Journal, vol. 97, no. 1162, pp. 571–2, 1995.
def h_peakcok(x, t):
    #x is the solid mass fraction
    #t is the temperature in ºC
    return 2.326 * (x / 10 * (100 + x) / (900 - 8 * x) + 1.8 * t * (1 - x / 10 * (0.6 - 0.0009 * t))) #in kJ/kg
#correlation that calculates the enthalpy of the sucrose-water mixture
#source: S. A. Nebra and M. I. Fernández-Parra, “THE EXERGY OF SUCROSE-WATER SOLUTIONS: 
#PROPOSAL OF A CALCULATION METHOD,” p. 9, 2005.
def h_kadlec(x, t, Pz):
    #x is the solid mass fraction
    #t is the temperature in ºC
    #Pz is the purity of solids present in the solution. Ex: Pz = 100% when there are only sucrose.
    return (4.1868 - 0.0297 * x + 4.6 / 100000 * x * Pz) * t + 3.75 / 100000 * x * t ** 2 #in kJ/kg
#correlation that calculates the entropy of the sucrose-water mixture at t temperature (in celsius degrees)
#x=concentration of solid mass
#source: E. P. Gyftopoulos and G. P. Beretta, Thermodynamics: foundations and applications. 
#Courier Corporation, 2005.
#take a look at: S. A. Nebra and M. I. Fernández-Parra, “THE EXERGY OF SUCROSE-WATER SOLUTIONS: 
#PROPOSAL OF A CALCULATION METHOD,” p. 9, 2005.
def delta_s(x, Pz, T, T_0):
    #x is the solid mass fraction
    #T is the temperature in K
    #Pz is the purity of solids present in the solution. Ex: Pz = 100% when there are only sucrose.
    #0 index -> dead state
    return (4.1868 - 0.05018625 * x + 4.6 / 100000 * x * Pz) * np.log(T / T_0) + 7.5 / 100000 * x * (T - T_0) #in kJ
#function that calculates the activity of water in sucrose-water solutions
#source: M. Starzak and S. D. Peacock, “Water activity coefficient in aqueous solutions of sucrose-a 
#comprehensive data analysis,” Zuckerindustrie, vol. 122, no. 5, pp. 380–388, 1997.
def activity_water(y_w, y_suc, T):
    #y_suc is the sucrose molar fraction
    #T is the temperature in K
    return y_w * np.exp(-2121.4052 / T * y_suc ** 2 * (1 - 1.0038 * y_suc - 0.24653 * y_suc ** 2))
#function that calculates the activity of sucrose in sucrose-water solutions
#source: S. A. Nebra and M. I. Fernández-Parra, “THE EXERGY OF SUCROSE-WATER SOLUTIONS: 
#PROPOSAL OF A CALCULATION METHOD,” p. 9, 2005.
def activity_sucrose(y_suc, x_suc_sat, T):
    #y_suc is the sucrose molar fraction
    #x_suc_sat is the saturated sucrose mass fraction
    #T is the temperature in K
    a_suc_sat = 1 / mass_to_molar_frac( #calculates the activity of saturated sucrose in sucrose-water solution
        x_suc_sat / 100, 
        prop_cp['sucrose']['molar_mass'], 
        prop_cp['water']['molar_mass']
    )
    return y_suc * a_suc_sat * np.exp(2121.4052 / T * (2 * y_suc + (-3 * 1.0038 - 2) / 2 * y_suc ** 2 + (-4 * 0.24653 + 3 * 1.0038) / 3 * y_suc ** 3 - 0.24653 * y_suc ** 4))

#function that calculates the saturated solid concentration in sucrose-water solutions
#source: S. Peacock, “Predicting physical properties of factory juices and syrups,” 
#International Sugar Journal, vol. 97, no. 1162, pp. 571–2, 1995.
def x_sol_peakcock(t):
    #t is the temperature in ºC 
    return 63.753 + 0.13542 * t + 0.0008869 * t ** 2 - 2.222 / 1000000 * t ** 3
#function that calculates the variation of the sugarcane juice boiling temperature in ºC
#source: S. Peacock, “Predicting physical properties of factory juices and syrups, 
#”International Sugar Journal, vol. 97, no. 1162, pp. 571–2, 1995.
def delta_t_peackok(t, x):
    #t is the temperature in ºC
    #x is the mass fraction in %
    return 6.064 / 100000 * (
        (273 + t) ** 2 * x **2 / (374 - t) ** 0.38
        ) * (5.84 / 10000000 * (x - 40) ** 2 + 0.00072)
#function that calculates the quantity in kg to oxidyze a given organic substance
#reference in source: J. R. Baker, M. W. Milke, and J. R. Mihelcic, “Relationship between chemical and 
#theoretical oxygen demand for specific classes of organic chemicals,” Water Research, vol. 33, no. 2, pp. 327–334, Feb. 1999, doi: 10.1016/S0043-1354(98)00231-0.
#take a look at: R. N. Nakashima and S. de Oliveira Junior, “Comparative exergy assessment of vinasse 
#disposal alternatives: Concentration, anaerobic digestion and fertirrigation,” Renewable Energy, vol. 147, pp. 1969–1978, Mar. 2020, doi: 10.1016/j.renene.2019.09.124.
def oxi_cod(c, h, o, s, n, q_o2, mm_s):
    #c = carbon; h = hydrogen; o = oxygen; s = sulphur; n = nitrogen
    return q_o2 / ((4 * c + h - 3 * n - 2 * o + 6 * s) / 4 * cp('M', 'T', t0, 'P', p0, 'oxygen') / mm_s)
#function that calculates the production of methane from vinasse
#source: C. A. de Lemos Chernicharo, Anaerobic reactors. London: IWA Publ. [u.a.], 2007.
def v_ch4(p, T, n, y_meth, y_acid, t_so4, cod):
    #p = pressure in the anaerobic reactor (Pa)
    #T = temperature in the anaerobic reactor (K)
    #n = removal efficiency of organic matter
    #y_meth = yield coefficient for acidogenic
    #y_acid = yield coefficient for methanogenic microorganisms
    #t_so4 = sulfate concentration in vinasse in kg/m3
    cod_r = cod * (1 - t_so4 * 2 / 3) * (1 - (y_meth - y_acid * (1 - y_meth)))
    #cod_r is the cod avaiable for methane production. t_so4 * 2 / 3 means that each 1.5 kg of SO4 that is produced, 1 kg of oxygen is needed
    return n * 1000* R * T * cod_r / (0.064 * p) # 0.064 means that 0.064 kg of oxygen is needed for 1 mol of CH4 that is produced
#function that calculates the nutritional requirement of an methanogenic bacteria
#Source: C. A. de Lemos Chernicharo, Anaerobic reactors. London: IWA Publ. [u.a.], 2007.
def nutritional_req(cod, vss, tss, n_bac):
    #vss = volatile sus
    #tss = total suspended solids
    #n_bac = concentration of nutrient in the bacterial cell (g/gVSS)
    y = vss / cod
    return cod * y * n_bac * tss / vss
#function that calculates the mass of water needed in the vaccum system from P. Rein, 
#Cane sugar engineering, Second edition. Berlin: Verlag Dr. Albert Bartens KG, 2017. (page 321)
def mass_vaccum(m_steam, t_cond_in, t_cond_out, alpha_vac):
    #m_steam = steam mass flow that enters the vaccum system
    #t_cond_in = temperature of the condensate that enters the vaccum system
    #t_cond_out = temperature of the condensate that leaves the vaccum system
    #alpha_vac = safety factor of the vaccumm system
    return m_steam *  570 / ((t_cond_out - t_cond_in) * (1 + alpha_vac))
#function that calculates the brix of the juice that enters the decanter
def int_brix(brix, e):
    h = h_kadlec(brix * 100, temperature(temperature_2[8], 'c'), parameter_1[1] * 100)
    if e < 0.00001:
        return brix
    else:
        m_vap = m_2_dosed_juice * ((1 - h_2_djuice105 / h) / (1 - (cp('H', 'T', temperature_2[8], 'Q', 1, fluid) / 1000) / h))
        m_juice_out = m_2_dosed_juice - m_vap
        brix_new = m_2_dosed_juice*brix_1_juice/m_juice_out
        e = brix_new-brix
        if e < 0:
            e = e * -1
        return int_brix(brix_new, e)

#####################################################################################################
######################################## GLOBAL VARIABLES ###########################################
#####################################################################################################
t0, p0, lw, R, CODv, m_sc, ex_sc = temperature(25, 'k'), pressure(1.01325, 'pa'), (cp('H', 'P', pressure(1.01325, 'pa'), 'Q', 1, 'water') - cp('H', 'P', pressure(1.01325, 'pa'), 'Q', 0, 'water')) / 1000, cp('GAS_CONSTANT', 'T', temperature(25, 'k'), 'P', pressure(1.01325, 'pa'), 'air') / 1000, 30, mass_flow(500, 'kg'), 5297
#t0 = temperature of the dead state in K
#p0 = pressure of the dead state in Pa
#lw = heat value for a p0 in kJ/kg
#R = molar gas constant in kJ/molK
#CODv = chemical oxygen demand of vinasse in kg/m^3
#m_sc = sugarcane crushing capacity in kg/s
#ex_sc = exergetic content of sugarcane in kJ/kg. Source: A. V. Ensinas, M. Modesto, S. A. Nebra, 
#and L. Serra, “Reduction of irreversibility generation in sugar and ethanol production from sugarcane,” Energy, vol. 34, no. 5, pp. 680–688, May 2009, doi: 10.1016/j.energy.2008.06.001.

#chemical exergies in kJ/kg, molar_mass in kg/mol, and enthalpy of formation in kJ/mol
#chemical exergies from Szargut, J., Morris, D.R., Steward, F.R., 1987. Exergy Analysis of Thermal, Chemical,
#and Metallurgical Processes. Hemisphere Publishing, New York.
prop_cp = {
    'water': {
        'chem_x': 49.95762483244768,
        'molar_mass' :0.01801528,
        'enthalpy_f': -241.8 #(g)
    },  
    'sucrose': {
        'chem_x': 17551.26671311324,
        'molar_mass': 0.34229648,
        'enthalpy_f': -2221.2
    },
    'd-fructose': {
        'chem_x': 15504.196101992427,
        'molar_mass': 0.18016,
        'enthalpy_f': 0
    },
    'cellulose': {
        'chem_x': 20997,
        'molar_mass': 0.3423,
        'enthalpy_f': - 983 #Source: BLOKHIN, A. V. et al. Thermodynamic Properties of Plant Biomass Components. 
        #Heat Capacity, Combustion Energy, and Gasification Equilibria of Cellulose. Journal of Chemical & Engineering Data, v. 56, n. 9, p. 3523–3531, 8 set. 2011. 
    },
    'hemicellulose': {
        'chem_x': 21395,
        'molar_mass': 0.1621406,
        'enthalpy_f': - 983 #Source: BLOKHIN, A. V. et al. Thermodynamic Properties of Plant Biomass Components. 
        #Heat Capacity, Combustion Energy, and Gasification Equilibria of Cellulose. Journal of Chemical & Engineering Data, v. 56, n. 9, p. 3523–3531, 8 set. 2011. 
    },
    'lignin': {
        'chem_x': 28161,
        'molar_mass': 0.194100167,
        'enthalpy_f': - 712.9 #Source: VOITKEVICH, O. V. et al. Thermodynamic properties of plant biomass components. 
        #Heat capacity, combustion energy, and gasification equilibria of lignin. Journal of Chemical & Engineering Data, v. 57, n. 7, p. 1903–1909, 2012. 
    },
    'lime': {
        'chem_x': 1965.1094986830915,
        'molar_mass': 0,
        'enthalpy_f': 0
    },
    'sulfuric_acid': {
        'chem_x': 1666,
        'molar_mass': 0,
        'enthalpy_f': 0
    },
    'sulfur_dioxide': {
        'chem_x': 4892,
        'molar_mass': 0,
        'enthalpy_f': 0
    },
    'air': {
        'chem_x': 23.821475647201872,
        'molar_mass': 0.02896546,
        'enthalpy_f': 0
    },
    'nitrogen': {
        'chem_x': 24.988,
        'molar_mass': 0.0280134,
        'enthalpy_f': 0
    },
    'oxygen': {
        'chem_x': 121.8796,
        'molar_mass': 0.0319988,
        'enthalpy_f': 0
    },
    'carbon_dioxide': {
        'chem_x': 422.6317,
        'molar_mass': 0.0440095,
        'enthalpy_f': -393.474
    },
    'methane': {
        'chem_x': 51820.4489,
        'molar_mass': 0.01604,
        'enthalpy_f': -74.8
    },
    'hydrogen_sulfide': {
        'chem_x': 23826.2911,
        'molar_mass': 0.03408,
        'enthalpy_f': 0
    },
    'bicarbonate': {
        'chem_x': 257.1211,
        'molar_mass': 0.0840071,
        'enthalpy_f': 0
    },
    'sodium_hydroxide': {
        'chem_x': 1872.6311,
        'molar_mass': 0.0399972,
        'enthalpy_f': 0
    },
    'sodium_sulfide': {
        'chem_x': 11806.1606,
        'molar_mass': 0.078044,
        'enthalpy_f': 0
    },
    'calcium_oxide': {
        'chem_x': 2269.9715,
        'molar_mass': 0.05608,
        'enthalpy_f': 0
    },
    'ethanol': {
        'chem_x': 29453.3259,
        'molar_mass': 0.0460695,
        'enthalpy_f': 0
    }
}
#sugarcane composition in mass fraction
#source: R. Palacios-Bereche et al., “Exergetic analysis of the integrated first- and 
#second-generation ethanol production from sugarcane,” Energy, vol. 62, pp. 46–61, Dec. 2013, doi: 10.1016/j.energy.2013.05.010.
sugarcane = {
    'water': [0.6935],
    'sucrose': [0.1385],
    'fibres': [0.1315, {
        'cellulose': [0.32],
        'hemicellulose': [0.2],
        'lignin': [0.18]
    }
    ],
    'd-fructose': [0.0059],
    'potassium_oxide': [0.002], #representing minerals
    'non_sacharides': [0.0179, {
        'aconitic_acid': [0.5],
        'potassium_chloride': [0.5]
    }
    ],
    'silicon_dioxide': [0.0107] # representing the soil    
}
#bagasse composition in mass fraction
#depends on the sugarcane fiber content and moisture fraction
#source: R. Palacios-Bereche et al., “Exergetic analysis of the integrated first- and 
#second-generation ethanol production from sugarcane,” Energy, vol. 62, pp. 46–61, Dec. 2013, doi: 10.1016/j.energy.2013.05.010.
bagasse = {
    'water': 0.5,
    'fibres': [0.47, {
        'cellulose': 0.39,
        'hemicellulose': 0.37,
        'lignin': 0.24
    }
    ],
    'ash': 0.03
}
#bagasse atoms composition in fraction
bagasse_atom = {
    'carbon': 0.4548,
    'hydrogen': 0.0596,
    'oxygen': 0.4521,
    'nitrogen': 0.0015
}
#air composition in fraction
air = {
    'nitrogen': 0.78084,
    'oxygen': 0.209476,
    'argon': 0.009364,
    'carbon dioxide': 0.000320
}
#vinasse composition
#COD fractions were adapted from E. L. Barrera, H. Spanjers, K. Solon, Y. Amerlinck, I. Nopens, and J. Dewulf,
#“Modeling the anaerobic digestion of cane-molasses vinasse: Extension of the Anaerobic Digestion Model No. 1 (ADM1) with sulfate reduction for a very high strength and sulfate rich wastewater,” Water Research, vol. 71, pp. 42–54, Mar. 2015, doi: 10.1016/j.watres.2014.12.026.
#The composition of vinasse was model taking into account the work of R. N. Nakashima and S. de Oliveira Junior, 
#“Comparative exergy assessment of vinasse disposal alternatives: Concentration, anaerobic digestion and fertirrigation,” Renewable Energy, vol. 147, pp. 1969–1978, Mar. 2020, doi: 10.1016/j.renene.2019.09.124.
#take a look at: R. N. Nakashima, “Avaliação exergética da geração e uso de biogás no setor sucroenergético.,”
#Mestrado em Engenharia Mecânica de Energia de Fluidos, Universidade de São Paulo, São Paulo, 2018. doi: 10.11606/D.3.2018.tde-27082018-153742.
vinasse = {
    'organic': { #COD fractions in gO2/gO2 and molar mass in kg/mol
        'sugars': { #Dextrose (C_6_H_12_O_6)
            'x_cod': 0.518,
            'c': 6,
            'h': 12,
            'o': 6,
            's': 0,
            'n': 0,
            'molar_mass': 0.18116,
        },
        'amino_acids': { #Average amino acid (C_4.82_H_8.95_O_2.88_N_1.13_S_0.02)
            'x_cod': 0.089, 
            'c': 4.82,
            'h': 8.95,
            'o': 2.88,
            's': 0.02,
            'n': 1.13,
            'molar_mass':0.129461942
        }, 
        'fatty_acids': { #Linoleic acid (C_18_H_32_O_2)
            'x_cod': 0.001,
            'c': 18,
            'h': 32,
            'o': 2,
            's': 0,
            'n': 0,
            'molar_mass': 0.280452
        },
        'acetic_acid': { #Acetic acid (C_2_H_4_O_2)
            'x_cod': 0.021, 
            'c': 2,
            'h': 4,
            'o': 2,
            's': 0,
            'n': 0,
            'molar_mass': 0.06005196
        },
        'carbohydrates': { #Sucrose (C_12_H_22_O_11)
            'x_cod': 0.106,
            'c': 12,
            'h': 22,
            'o': 11,
            's': 0,
            'n': 0,
            'molar_mass': 0.34229648
        },  
        'proteins': { #Average amino acid (C_4.82_H_8.95_O_2.88_N_1.13_S_0.02)
            'x_cod': 0.001, 
            'c': 4.82,
            'h': 8.95,
            'o': 2.88,
            's': 0.02,
            'n': 1.13,
            'molar_mass': 0.129461942
        }, 
        'organic_inert': { #Lignin (C_7.3_H_13.9_O_1.3)
            'x_cod': 0.261, 
            'c': 7.3,
            'h': 13.9,
            'o': 1.3,
            's': 0,
            'n': 0,
            'molar_mass': 0.122493
        }
    },
    'inorganic': { #concentration in kg/m3 and chemical exergy in kJ/mol
        'nitrogen': { #NH_3
            'x': 0.037,
            'molar_mass': 0.01703052,
            'chem_ex': 337.9
        },  
        'phosphorus': { #H_3_PO_4
            'x': 0.034,
            'molar_mass': 0.097995181,
            'chem_ex': 89.6
        },
        'potassium': { #KOH and KCl
            'x': 2.206,
            'molar_mass':0.05610564,
            'chem_ex': 63.6 #median between KOH and KCl
        },
        'calcium': { #Ca(OH)_2
            'x': 0.832,
            'molar_mass': 0.07409268,
            'chem_ex': 70.8
        },
        'magnesium': { #Mg(OH)_2
            'x': 0.262,
            'molar_mass': 0.05831968,
            'chem_ex': 33.2
        },
        'sulfate': { #H_2_SO_4
            'x': 1.149,
            'molar_mass': 0.09807848,
            'chem_ex': 163.4
        },
        'chlorine': { #KCl
            'x': 1.219,
            'molar_mass': 0.0745513,
            'chem_ex': 19.6
        }
    }
}
#composition of biogas from K. R. Salomon, E. E. S. Lora, M. H. Rocha, and O. O. Almazán, 
#“Cost calculations for biogas from vinasse biodigestion and its energy utilization,” Sugar industry, vol. 136, no. 4, pp. 217–223, 2011.
biogas = {
    'methane': 0.6,
    'carbon_dioxide': 0.39,
    'hydrogen_sulfide': 0.01
}
#chemical composition of the methanogenic microorganisms (g/kgTSS) from G. Lettinga, L. W. Hulshof Pol, and 
#G. Zeeman, “Biological wastewater treatment. Part I: Anaerobic wastewater treatment,” Lecture Notes. Wageningen Agricultural University, ed, 1996.
#take a look at: C. A. de Lemos Chernicharo, Anaerobic reactors. London: IWA Publ. [u.a.], 2007.
bacteria = { 
    'macronutrients': {
        'nitrogen': 65,
        'phosphorus': 15,
        'potassium': 10,
        'sulfate': 10,
        'calcium': 4,
        'magnesium': 3
    },
    'micronutrients': {
        'iron': 1.8,
        'nickel': 0.1,
        'cobalt': 0.075,
        'molybdenum': 0.06,
        'zinc': 0.06,
        'manganese': 20,
        'copper': 0.01
    }
}
#####################################################################################################
############################ CANE PREPARATION AND JUICE EXTRACTION (1) ##############################
#####################################################################################################
class Juice:
    def __init__(self, m:float, Pol:float, Pz:float, Brix:float, t:float) -> None:
        '''
        Class that defines the sucrose-water solution

        :param float m: juice mass flow.
        :param float Pol: the Pol of the solution (fraction).
        :param float Pz: the purity of solids present in the solution. Ex: Pz = 100% when there are only sucrose.
        :param float Brix: the solid mass fraction
        :param float t: the temperature
        '''
        self.m = m
        self.Pol = Pol
        self.ysucrose = mass_to_molar_frac(Pol, prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass'])
        self.ywater = 1 - self.ysucrose
        self.Pz = Pz
        self.Brix = Brix
        self.xwater = None
        self.t = t
        pass

    def x_sol_peakcock(self):
        '''
        Calculates the saturated solid concentration in sucrose-water solutions. Source: S. Peacock, “Predicting physical properties of factory juices and syrups,” 
        International Sugar Journal, vol. 97, no. 1162, pp. 571–2, 1995.

        :return: saturated solid concentration (admensional)
        '''
        return 63.753 + 0.13542 * self.t + 0.0008869 * self.t ** 2 - 2.222 / 1000000 * self.t ** 3

    def delta_t_peackok(self):
        '''
        Calculates tthe variation of the sugarcane juice boiling temperature in C. Source: S. Peacock, “Predicting physical properties of factory juices and syrups,” 
        International Sugar Journal, vol. 97, no. 1162, pp. 571–2, 1995.

        :return: juice boiling temperature variation in C
        '''
        return 6.064 / 100000 * (
            (273 + self.t) ** 2 * self.Brix **2 / (374 - self.t) ** 0.38
            ) * (5.84 / 10000000 * (self.Brix - 40) ** 2 + 0.00072)
    
    def cp_kadlec(self):
        '''
        Correlation that calculates the specific heat of the sucrose-water mixture at "t" temperature in C. Source: KADLEC, P.; BRETSCHNEIDER, R.; DANDAR, A. Measurement and calculation of physical-chemical 
        properties of sugar solutiuons. Sucrerie belge, 1981.

        :return: the specific heat of the sucrose-water mixture in kJ/kgK
        '''
        return  4.868 - 0.0297 * self.Brix + 0.000046 * self.Brix * self.Pz + 0.000075 * self.Brix * self.t #in kJ/kgK
    def cp_suc(self):
        '''
        Correlation that calculates the specific heat of pure at "t" temperature in C. Source: ANDERSON JR, G. L.; HIGBIE, H.; STEGEMAN, G. The heat capacity of sucrose from 25 to 90. 
        Journal of the American Chemical Society, v. 72, n. 8, p. 3798–3799, 1950. 

        :param float m: juice mass flow.
        :param float Pol: the Pol of the solution (fraction).
        :param float Pz: the purity of solids present in the solution. Ex: Pz = 100% when there are only sucrose.
        :param float Brix: the solid mass fraction
        :param float t: the temperature
        :return: the specific heat of the pure sucrose in kJ/kgK
        '''
        return 1.244 + 4.819 / 1000 * (self.t - 25) + 6.238 / 1000000 * (self.t - 25) ** 2 
    
    def h_peakcok(self):
        '''
        Correlation that calculates thhe enthalpy of the sucrose-water mixture at "t" temperature in C. Source:S. Peacock, “Predicting physical properties of factory juices and syrups,” 
        International Sugar Journal, vol. 97, no. 1162, pp. 571–2, 1995.
        
        :return: the enthalpy of the sucrose-water mixture in kJ/kg
        '''
        return 2.326 * (self.Brix / 10 * (100 + self.Brix) / (900 - 8 * self.Brix) + 1.8 * self.t * (1 - self.Brix / 10 * (0.6 - 0.0009 * self.t)))
    
    def h_kadlec(self):
        '''
        Correlation that calculates the enthalpy of the sucrose-water mixture at "t" temperature in C. Source: KADLEC, P.; BRETSCHNEIDER, R.; DANDAR, A. Measurement and calculation of physical-chemical 
        properties of sugar solutiuons. Sucrerie belge, 1981.

        :return: the enthalpy of the sucrose-water mixture in kJ/kg
        '''
        h_sugar = self.cp_suc() * temperature(self.t, 'k')
        x_sol = self.x_sol_peakcock()
        h_result = (4.1868 - 0.0297 * self.Brix + 4.6 / 100000 * self.Brix * self.Pz) * self.t + 3.75 / 100000 * self.Brix * self.t ** 2 
        if self.Brix > x_sol:
            z =  (100 - self.Brix) / (100-x_sol)
        else:
            z = 1
        return z * h_result + (1 - z) * h_sugar
    
    def delta_s(self):
        '''
        Correlation that calculates the entropy of the sucrose-water mixture at "t" temperature in C. Source: E. P. Gyftopoulos and G. P. Beretta, Thermodynamics: foundations and applications. 
        Courier Corporation, 2005. Take a look at: S. A. Nebra and M. I. Fernández-Parra, “THE EXERGY OF SUCROSE-WATER SOLUTIONS: PROPOSAL OF A CALCULATION METHOD,” p. 9, 2005.
        
        :return: the entropy of the sucrose-water mixture in kJ/kgK
        '''
        T = temperature(self.t, 'k')
        T_0 = t0 #dead-state temperature
        return (4.1868 - 0.05018625 * self.Brix + 4.6 / 100000 * self.Brix * self.Pz) * np.log(T / T_0) + 7.5 / 100000 * self.Brix * (T - T_0)
    
    def activity(self, calculate:str):
        '''
        Calculates the activity of sucrose or water in sucrose-water solutions. Source: M. Starzak and S. D. Peacock, “Water activity coefficient in aqueous solutions of sucrose-a 
        comprehensive data analysis,” Zuckerindustrie, vol. 122, no. 5, pp. 380–388, 1997.
        
        :param str calculate: "water" to calculate water activity or "sucrose" to calculate sucrose activity
        :return: the activity of sucrose or water in sucrose-water solutions (admensional)
        '''
        A = - 1.0038
        B = - 0.24653
        T = temperature(self.t, 'k') #temperature in K
        
        if calculate.lower() == 'water':
            a = self.ywater * np.exp(-2121.4052 / T * self.ysucrose ** 2 * (1 + A * self.ysucrose + B * self.ysucrose ** 2))
        elif calculate.lower() == 'sucrose':
            y_suc_sat = mass_to_molar_frac(self.x_sol_peakcock() / 100, prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass'])
            a = self.ysucrose / y_suc_sat * np.exp(- 2121.4052 / T * (2 * (y_suc_sat - self.ysucrose) + (3 * A - 2) / 2 * (y_suc_sat ** 2 - self.ysucrose ** 2) + (4 * B - 3 * A) / 3 * (y_suc_sat ** 3 - self.ysucrose ** 3) - B * (y_suc_sat ** 4 - self.ysucrose ** 4)))
        return a

class LignoCel:
    def __init__(self, xashes:float, xwater:float, xfiber:float, xcell:float, xhemi:float, xlign:float, xsucrose:float, m:float = 1) -> None:
        '''
        Class that describes a LignoCellulosic material

        :param float xashes: mass fraction of ashes in the material.
        :param float xwater: mass fraction of water in the material.
        :param float xfiber: mass fraction of fibers in the material.
        :param float xcell: mass fraction of cellulose in the material's fiber.
        :param float xhemi: mass fraction of hemicellulose in the material's fiber.
        :param float xlignin: mass fraction of lignin in the material's fiber.
        :param float xsucrose: mass fraction of sucrose in the material.
        :param float m: material mass fraction in kg/s
        '''
        xtotaldry = xfiber + xashes + xsucrose
        xfibern = xfiber / xtotaldry #mass fraction of fiber in dry basis
        xashesn = xashes / xtotaldry #mass fraction of ashes in dry basis
        xsucrosen = xsucrose / xtotaldry #mass fraction of sucrose in dry basis

        self.m = { #material's mass flow in a wet and dry basis (kg/s)
            'wet': m,
            'dry': m * (1 - xwater)
        }
        self.MM = { #lignocellulosic material's molar mass in a wet and dry basis
            'wet': 1 / (xfiber * (xcell / prop_cp['cellulose']['molar_mass'] + xhemi / prop_cp['hemicellulose']['molar_mass'] + xlign / prop_cp['lignin']['molar_mass']) + xsucrose / prop_cp['sucrose']['molar_mass'] + xashes / prop_cp['sodium_sulfide']['molar_mass'] + xwater / prop_cp['water']['molar_mass']),
            'dry':1 / (xfibern * (xcell / prop_cp['cellulose']['molar_mass'] + xhemi / prop_cp['hemicellulose']['molar_mass'] + xlign / prop_cp['lignin']['molar_mass']) + xsucrosen / prop_cp['sucrose']['molar_mass'] + xashesn / prop_cp['sodium_sulfide']['molar_mass'])
        }
        self.n = { #number of mols in the material in a wet and dry basis
            'wet': m / self.MM['wet'], 
            'dry': m * (1 - xwater) / self.MM['dry']
        }
        self.xashes = { #mass fraction of ashes in dry and wet basis
            'wet': xashes,
            'dry': xashesn
        }
        self.xwater = xwater #mass fraction of water
        self.xfiber = { #mass fraction of fiber in dry and wet basis
            'wet': xfiber,
            'dry': xfibern
        }
        self.xsucrose = { #mass fraction of sucrose in dry and wet basis
            'wet': xsucrose,
            'dry': xsucrosen
        }
        self.ysucrose =  { #molar fraction of sucrose in dry and wet basis
            'wet':  xsucrose * prop_cp['sucrose']['molar_mass'] / self.MM['wet'],
            'dry': xsucrosen * prop_cp['sucrose']['molar_mass'] / self.MM['dry']
        }
        self.compsucrose = { #composition of sucrose
            'C': 6,
            'H': 12,
            'O': 6
        }
        self.xcell = { #mass fraction of cellulose in dry and wet basis
            'wet': xfiber * xcell,
            'dry': xfibern * xcell
        }
        self.ycell = { #molar fraction of cellulose in dry and wet basis
            'wet': self.xcell['wet'] * prop_cp['cellulose']['molar_mass'] / self.MM['wet'],
            'dry': self.xcell['dry'] * prop_cp['cellulose']['molar_mass'] / self.MM['dry']
        }
        self.compcell = { #composition of cellulose
            'C': 6,
            'H': 10,
            'O': 5
        }
        self.xhemi = { #mass fraction of hemicellulose in dry and wet basis
            'wet': xfiber * xhemi,
            'dry': xfibern * xhemi
        }
        self.yhemi = { #molar fraction of hemicellulose in dry and wet basis
            'wet': self.xhemi['wet'] * prop_cp['hemicellulose']['molar_mass'] / self.MM['wet'],
            'dry': self.xhemi['dry'] * prop_cp['hemicellulose']['molar_mass'] / self.MM['dry']
        }
        self.comphemi = { #composition of hemicellulose
            'C': 6,
            'H': 10,
            'O': 5
        }
        self.xlign = { #molar fraction of lignin in dry and wet basis
            'wet': xfiber * xlign,
            'dry': xfibern * xlign
        }
        self.yhemi = { #molar fraction of lignin in dry and wet basis
            'wet': self.xlign['wet'] * prop_cp['lignin']['molar_mass'] / self.MM['wet'],
            'dry': self.xlign['dry'] * prop_cp['lignin']['molar_mass'] / self.MM['dry']
        }
        self.complign = { #composition of lignin
            'C': 10,
            'H': 11.5,
            'O': 3.9
        }

    def EnthalpyF(self, tp:str) -> float:
        '''
        Calculate the Enthalpy of formation of an lignocellulosic material in dry basis.

        :param str tp: use "spec_mol" for specific enthalpy of formation in kJ/mol; "spec_mass" for enthalpy of formation
        in kJ/kg; and "full" for kJ.
        :return:  Enthalpy of formation of an lignocellulosic material in dry basis. (kJ/mol, kJ/kg, or kJ).
        '''
        h = self.ycell['dry'] * prop_cp['cellulose']['enthalpy_f'] + self.yhemi['dry'] * prop_cp['hemicellulose']['enthalpy_f'] + self.ylign['dry'] * prop_cp['lignin']['enthalpy_f']
        if tp.lower() == 'spec_mol':
            pass
        elif tp.lower() == 'spec_mass':
            h *= self.n['dry'] / self.m['dry']
        elif tp.lower() == 'full':
            h *= self.n['dry']
        else:
            h = None
        return h
    def MolsAtom(self, at:str) -> float:
        '''
        Get the mols of a given element in the lignocel's material

        :param str at: 'C' for carbon, 'H' for hydrogen, and 'O' for oxygen.
        :return: mols of carbon, hydrogen or oxygen within an lignocel's material.
        '''
        return self.n['dry'] * (self.ycell['dry']  * self.compcell[at.upper()] + self.yhemi['dry']  * self.comphemi[at.upper()] + self.ylign['dry']  * self.complign[at.upper()] + self.sucrose['dry'] * self.compsucrose[at.upper()])
    def GetElementPercentage(self, at:str) -> float:
        '''
        Get the mol fraction of a given element in the material

        :param str at: 'C' for carbon, 'H' for hydrogen, and 'O' for oxygen.
        :return: mol fraction of carbon, hydrogen or oxygen within an lignocel's material.
        '''
        return self.MolsAtom(at) / self.n['dry']
    def getWater(self, tp:str) -> float:
        '''
        Get the amount of water in the material in kg or mols.

        :param str tp: 'mol' for mols of water, and 'mass' for kg of water.
        :return: the amount of water in the material in kg or mols
        '''
        if tp.lower() == 'mol':
            return self.m['wet'] * self.xwater / prop_cp['water']['molar_mass']
        elif tp.lower() == 'mass':
            return self.m['wet'] * self.xwater
    def HHV(self) -> float:
        '''
        Calculates the HHV content of fuels in MJ/kg. source: S. A. Channiwala and P. P. Parikh, “A unified correlation for estimating HHV of solid, 
        liquid and gaseous fuels,” Fuel, vol. 81, no. 8, pp. 1051–1063, May 2002, doi: 10.1016/S0016-2361(01)00131-4.

        :return: the HHV content of the material in MJ/kg.
        '''
        c = self.GetElementPercentage('C') * 100 #c is the carbon content expressed in %
        h = self.GetElementPercentage('H') * 100 #h is the hydrogen content expressed in %
        s = 0 #s is the sulfur content expressed in %
        o = self.GetElementPercentage('O') * 100 #o is the oxygen content expressed in %
        n = 0 #n is the nitrogen content expressed in %
        a = self.xashes['dry'] * 100 #a is the ash content expressed in %
        return (0.3491 * c + 1.1783 * h + 0.1005 * s - 0.1034 * o - 0.0151 * n - 0.0211 * a)
    def LHV(self):
        '''
        Calculates the LHV content of fuels in kJ/kg. Source: W. A. Bizzo and C. G. Sánchez, “Stoichiometry of combustion and gasification reactions 
        (Estequiometria das reações de combustão e gaseificação in Portuguese),” Biomass Gasification Technology (Tecnologia Da Gaseificação de Biomassa in Portuguese), p. 430, 2010.
        
        :return:the LHV content of the material in kJ/kg.
        '''
        return (1000 * self.HHV() - lw * (9 * self.GetElementPercentage('H') - self.xwater)) * (1 - self.xwater)
    def beta_fuel(self) -> float:
        '''
        Calculates beta index of the chemical exergy of solid fuels. Source: Szargut, J., Morris, D.R., Steward, F.R., 1987. Exergy Analysis of Thermal, Chemical,
        #and Metallurgical Processes. Hemisphere Publishing, New York.

        :return: beta index of the chemical exergy of solid fuels (admensional)
        '''
        z_h = self.GetElementPercentage('H') #z_h is the hydrogen content expressed in decimals
        z_c = self.GetElementPercentage('C') #z_c is the carbon content expressed in decimals
        z_o = self.GetElementPercentage('O') #z_o is the oxygen content expressed in decimals
        z_n = 0 #z_n is the nitrogen content expressed in decimals
        return (1.0412 + 0.216 * z_h / z_c - 0.2499 * z_o / z_c * (1 + 0.7884 * z_h / z_c) + 0.045 * z_n / z_c) / (1 - 0.3035 * z_o / z_c)
    def Exergy(self) -> float:
        '''
        Calculates the sum of the standart chemical exergy in kJ/kg. Source: Szargut, J., Morris, D.R., Steward, F.R., 1987. Exergy Analysis of Thermal, Chemical,
        and Metallurgical Processes. Hemisphere Publishing, New York.

        :return: standart chemical exergy in kJ/kg.
        '''
        return self.m * (self.LHV() * beta_fuel() + prop_cp['water']['chem_x'] * self.xwater)

class PreparationSubsystem:
    def __init__(self, sugarcane, PI) -> None:
        self.sugarcane = sugarcane #class LigcelullosicMaterial
        self.PI = PI #preparation index
    def Prepare(self):
        HeavyDutyShredder = np.e ** (np.log(self.sugarcane.m * (1 - self.sugarcane.xwater - self.sugarcane.xashes)) + np.log(self.PI / 67.3) / 0.09)
        return HeavyDutyShredder + 30 + 75 + 75

class MillTandem:
    def __init__(self, sugarcane, efficiency) -> None:
        self.sugarcane = sugarcane
        self.efficiency = efficiency
        self.imbibition = None
        self.bagasse = None
        self.juice = None
        pass
    def Crush(self, xwater, xashes):
        self.bagasse = LignoCel(xashes, xwater, 1 - (xwater + xashes), self.sugarcane.xcell, self.sugarcane.xhemi, self.sugarcane.xlign, self.sugarcane.m * self.sugarcane.xfiber / (1 - (xwater + xashes)))
        self.imbibition = 2 * self.sugarcane.m * self.sugarcane.xfiber

        return None 

fluid = 'water'
parameter_1 = [ ### JUICE PARAMETERS ###
    0.97,   #[0] sugar extraction efficiency (n_extraction). Source: A. V. Ensinas, “Integração termica e otimização 
    #termoeconomica aplicadas ao processo industrial de produção de açucar e etanol a partir da cana-de-açucar,” Mar. 2018, Accessed: Aug. 11, 2021. [Online]. Available: http://repositorioslatinoamericanos.uchile.cl/handle/2250/1333309
    0.86,   #[1] juice purity (Pz). Source: A. V. Ensinas, M. Modesto, S. A. Nebra, and L. Serra, 
    #“Reduction of irreversibility generation in sugar and ethanol production from sugarcane,” Energy, vol. 34, no. 5, pp. 680–688, May 2009, doi: 10.1016/j.energy.2008.06.001.
    0.6972, #[2] lambda sugar -> juice fraction that goes to the sugar production. This parameter 
    #is adjusted in order to comply with a sugar to ethanol production ratio of 2.2. Source: M. Morandin, 
    #A. Toffolo, A. Lazzaretto, F. Maréchal, A. V. Ensinas, and S. A. Nebra, “Synthesis and parameter 
    #optimization of a combined sugar and ethanol production process integrated with a CHP system,” Energy, vol. 36, no. 6, pp. 3675–3690, 2011.
    ### OPTIMIZED PREPARATION POWER DEMAND ### 
    #Source: S. T. Inskip, “Cane preparation - optimised technology,” INTERNATIONAL SUGAR JOURNAL, 
    #vol. 112, no. 1339, p. 7, 2010.
    75,     #[3] carder drum power in kW
    75,     #[4] feeder drum power in kW
    30,     #[5] shedder kicker power in kW
    0.05,   #[6] shedder rotor power per kg of fiber (kJ/kg)
    ### ELETRIFIED MILL POWER DEMAND ###
    0.015,  #[7] mill's power per kg of sugarcane (kJ/kg)
]
temperature_1 = [ ### Source: A. V. Ensinas, “Integração termica e otimização termoeconomica aplicadas ao processo industrial de 
#produção de açucar e etanol a partir da cana-de-açucar,” Mar. 2018.
    35,                  #[0] temperature of the outflow raw juice in C
    temperature(98, 'k') #[1] temperature of the imbibition water in K
]
pol_1_sc = sugarcane['sucrose'][0] / (sugarcane['sucrose'][0] + sugarcane['d-fructose'][0] + sugarcane['potassium_oxide'][0] + sugarcane['silicon_dioxide'][0]) 
m_imbibition = 2 * sugarcane['fibres'][0] * m_sc #quantity of fiber in sugarcane in kg/s X 2
m_bagasse = sugarcane['fibres'][0] * m_sc / (1 - bagasse['water']) #bagasse mass flow in kg/s
m_rawjuice = m_sc - m_bagasse + m_imbibition #mass flow of raw juice in kg/s
pol_1_juice = m_sc * sugarcane['sucrose'][0] * parameter_1[0] / m_rawjuice #pol of juice
brix_1_juice = pol_1_juice / parameter_1[1]
pol_1_bagasse = (m_sc * sugarcane['sucrose'][0] - m_rawjuice * pol_1_juice) / m_bagasse #pol of bagasse
y_sucrose = mass_to_molar_frac(pol_1_juice, prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass']) #molar fraction of sucrose in raw juice
y_water = 1 - y_sucrose #molar frac of water in raw juice
h_mix = h_kadlec(brix_1_juice * 100, temperature_1[0], parameter_1[1] * 100) #enthalpy of water-sucrose mixture
delta_s_mix = delta_s(pol_1_juice * 100, pol_1_sc * 100, temperature(temperature_1[0], 'k'), t0) #delta entropy of water-sucrose mixture
h0_mix = h_kadlec(brix_1_juice * 100, temperature(t0, 'c'), parameter_1[1] * 100) #dead state enthalpy of water-sucrose mixture
w_1 = [ #list of exergies from work
    -(
        parameter_1[3] + parameter_1[4] + parameter_1[5] + m_sc * (sugarcane['fibres'][0] * parameter_1[6] + parameter_1[7])
    )
]


m_ex_1 = [ #list of exergies from mass flow
    m_sc * ex_sc, #[0] inflow sugarcane
    - m_bagasse * ch_solid_exergy( #[1] outflow bagasse
        lhv(
            1000 * hhv( 
                bagasse_atom['carbon'] * 100, 
                bagasse_atom['hydrogen'] * 100, 
                0, 
                bagasse_atom['oxygen'] * 100,
                bagasse_atom['nitrogen'] * 100, 
                0
                ),
                bagasse_atom['hydrogen'],
                bagasse['water']
                ),
                beta_fuel(
                    bagasse_atom['hydrogen'], 
                    bagasse_atom['carbon'], 
                    bagasse_atom['oxygen'], 
                    bagasse_atom['nitrogen']
                    ),
                    0,
                    0,
                    0,
                    0,
                    0, 
                    prop_cp['water']['chem_x'],
                    bagasse['water']
                    ),
    - m_rawjuice * (mix_exergy( #[2] outflow juice
        activity_sucrose(
            y_sucrose,
            x_sol_peakcock(temperature_1[0]),
            temperature(temperature_1[0], 'k')
            ),
            activity_water(
                y_water,
                y_sucrose,
                temperature(temperature_1[0], 'k')
                ),
                prop_cp['sucrose']['molar_mass'] * y_sucrose + prop_cp['water']['molar_mass'] * y_water,
                y_sucrose, 
                y_water,
                t0
                )+ph_exergy(
                    h_mix, 
                    delta_s_mix, 
                    h0_mix, 
                    0
                    ) + prop_cp['water']['chem_x'] * (1 - pol_1_juice) + prop_cp['sucrose']['chem_x'] * pol_1_juice
    ),
    m_imbibition * ph_exergy( #[3]
        cp('H', 'T', temperature_1[1], 'P', p0, fluid) / 1000,
        cp('S', 'T', temperature_1[1], 'P', p0, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']
]

print('Subsystem 1 - CANE PREPARATION AND JUICE EXTRACTION - exergy destruction: {:.2f} MW'.format(exergy_destruction(0, w_1, m_ex_1) / 1000))
#####################################################################################################
############################## JUICE TREATMENT TO THE SUGAR PRODUCTION (2) ##########################
#####################################################################################################
parameter_2 = [
    0.6972,  #[0] fraction of raw juice that goes to the sugar production. Source: ENSINAS, A. V. 
    #Integração termica e otimização termoeconomica aplicadas ao processo industrial de produção de açucar e etanol a partir da cana-de-açucar. 29 mar. 2018. 
    ### --> Source: PIZAIA, W.; NAKAHODO, T. O. DT Alternativas para redução do consumo de água no processo. 
    #Cooperativa de Produtores de Cana, Açúcar e álcool do Estado de São Paulo Ltda (Copersucar), 1999. 
    0.0006,  #[1] kgSO2/kg of sugarcane for sugar production. Source
    0.022,   #[2] kg/kg of sugarcane for sugar production -> water for sulfitation cooling
    0.00011, #[3] kgCaO/kg of sugarcane for sugar production
    0.0231,  #[4] kg/kg of sugarcane for sugar production -> water for lime production
    ### <-- Source: PIZAIA, W.; NAKAHODO, T. O. DT Alternativas para redução do consumo de água no processo. 
    #Cooperativa de Produtores de Cana, Açúcar e álcool do Estado de São Paulo Ltda (Copersucar), 1999.
    0.07,    #[5] kg/kg of sugarcane for sugar production -> water for cleaning the filter and sludge diluition. Source: Camargo, C.A. 
    #(coord.) Conservação de energia na indústria do açúcar e do álcool, Manual de recomendações. São Paulo: Instituto de Pesquisas Tecnológicas (IPT) (Publicação IPT; v.1871), 1990.
    0.0005,  #[6] kg of bagacilho/kg of sugarcane. Source: ENSINAS, A. V. Integração termica e otimização termoeconomica aplicadas ao processo industrial 
    #de produção de açucar e etanol a partir da cana-de-açucar. 29 mar. 2018. 
    ### ---> Source: Camargo, C.A. (coord.) Conservação de energia na indústria do açúcar e do álcool, Manual de recomendações. 
    #São Paulo: Instituto de Pesquisas Tecnológicas (IPT) (Publicação IPT; v.1871), 1990.    
    0.035,   #[7] kg of filter cake/ kg of sugarcane for sugar production.
    0.7,     #[8] filter cake moisture
    0.02,    #[9] filter cake POL
    ### <--- Source: Camargo, C.A. (coord.) Conservação de energia na indústria do açúcar e do álcool, Manual de recomendações. 
    #São Paulo: Instituto de Pesquisas Tecnológicas (IPT) (Publicação IPT; v.1871), 1990.
    0.10,    #[10] fraction of juice recirculated (related to the mix juice feeded). Source: ENSINAS, A. V. Integração termica e otimização termoeconomica aplicadas ao processo industrial 
    #de produção de açucar e etanol a partir da cana-de-açucar. 29 mar. 2018. 
    0.2      #[11] safety factor of the vaccumm system. Source: PIZAIA, W.; NAKAHODO, T. O. DT Alternativas para redução do consumo de água no processo. 
    #Cooperativa de Produtores de Cana, Açúcar e álcool do Estado de São Paulo Ltda (Copersucar), 1999.
]
temperature_2 = [
    ### ---> Source: PIZAIA, W.; NAKAHODO, T. O. DT Alternativas para redução do consumo de água no processo. 
    #Cooperativa de Produtores de Cana, Açúcar e álcool do Estado de São Paulo Ltda (Copersucar), 1999.
    temperature(85, 'k'),            #[0] temperature of the filtered juice in K
    temperature(70, 'k'),            #[1] temperature of the recirculated juice in K
    ### <--- Source: PIZAIA, W.; NAKAHODO, T. O. DT Alternativas para redução do consumo de água no processo. 
    #Cooperativa de Produtores de Cana, Açúcar e álcool do Estado de São Paulo Ltda (Copersucar), 1999.
    ### ---> Source: ENSINAS, A. V. Integração termica e otimização termoeconomica aplicadas ao processo industrial 
    #de produção de açucar e etanol a partir da cana-de-açucar. 29 mar. 2018.
    temperature(97, 'k'),            #[2] temperature of the juice that goes to the decanter in K
    temperature(35, 'k'),            #[3] temperature of the dosed juice in K
    temperature(105,'k'),            #[4] temperature of the heated juice in K
    temperature(97, 'k'),            #[5] temperature of the treated juice in K
    ### <--- Source: ENSINAS, A. V. Integração termica e otimização termoeconomica aplicadas ao processo industrial 
    #de produção de açucar e etanol a partir da cana-de-açucar. 29 mar. 2018.
    ### ---> Source: PIZAIA, W.; NAKAHODO, T. O. DT Alternativas para redução do consumo de água no processo. 
    #Cooperativa de Produtores de Cana, Açúcar e álcool do Estado de São Paulo Ltda (Copersucar), 1999.
    temperature(30, 'k'),            #[6] temperature of the inlet water that goes to the vaccumm system in K
    temperature(50, 'k'),            #[7] temperature of the outlet water that goes to the vaccumm system in K
    ### <--- Source: PIZAIA, W.; NAKAHODO, T. O. DT Alternativas para redução do consumo de água no processo. 
    #Cooperativa de Produtores de Cana, Açúcar e álcool do Estado de São Paulo Ltda (Copersucar), 1999.
    cp('T', 'Q', 1, 'P', p0, fluid), #[8] temperature of the saturated flashed steam
    temperature(107.4, 'k')          #[9] temperature of the water to dilute the lime in K. Source: ENSINAS, A. V. Integração termica e otimização termoeconomica aplicadas ao processo industrial 
    #de produção de açucar e etanol a partir da cana-de-açucar. 29 mar. 2018.
]
pressure_2 = [ ### Source: A. V. Ensinas, “Integração termica e otimização termoeconomica aplicadas ao processo industrial de 
#produção de açucar e etanol a partir da cana-de-açucar,” Mar. 2018.
    pressure(6, 'pa'),  #[0] pressure of the juice to the heating process in Pa
    p0,                 #[1] pressure of the juice treatment process in Pa
    pressure(2.5, 'pa') #[2] pressure of low-pressure steam line
]
m_2_juice = parameter_2[0] * m_rawjuice #mass flow of the treated juice in kg/s
m_2_rec_juice = parameter_2[10] * m_2_juice #mass flow of the recirculated juice in kg/s
m_2_SO2 = parameter_2[1] * m_2_juice #mass flow of SO2 in kg/s
m_2_wSO2 = parameter_2[2] * m_2_juice #mass flow of water for cooling the juice in kg/s
m_2_CaO = parameter_2[3] * m_2_juice #mass flow of CaO in kg/s
m_2_wCaO = parameter_2[4] * m_2_juice #mass flow of water for adding up to the CaO form lime in kg/s
m_2_dosed_juice = m_2_juice + m_2_SO2 + m_2_CaO + m_2_wCaO + m_2_rec_juice #juice mass flow that was 
#chemical treated in kg/s
### HEATING PROCESS###
h_2_whot = cp('H', 'Q', 1, 'P', pressure_2[2], fluid) / 1000 #saturated steam that heats the juice from liming
h_2_wcold = cp('H', 'Q', 0, 'P', pressure_2[2], fluid) / 1000 #saturated liquid that heated the juice from liming
h_2_djuice105 = h_kadlec(brix_1_juice * 100, temperature(temperature_2[4], 'c'), parameter_1[1] * 100) #enthalpy of outflow juice from the liming process
h_2_djuice35 = h_kadlec(brix_1_juice * 100, temperature(temperature_2[3], 'c'), parameter_1[1] * 100) # enthalpy of inflow juice from the liming process
m_2_wh = m_2_dosed_juice * (h_2_djuice105 - h_2_djuice35) / (h_2_whot - h_2_wcold) #mass flow of water to heat the juice up after the liming process          
brix_2_dec = int_brix(brix_1_juice, 1) #brix of the juice that enters the decanter
h_2_djuice_dec = h_kadlec(brix_2_dec * 100, temperature(temperature_2[8], 'c'), parameter_1[1] * 100) #enthalpy of the juice that enters the decanter
### FLASH TANK (ISOTHERMIC EXPANSION) ###
m_2_flashvap = m_2_dosed_juice * ( #saturated steam flashed in the flash tank
    (
        1 - h_2_djuice105 / h_2_djuice_dec
    ) / (1 - (
        cp('H', 'T', temperature_2[8], 'Q', 1, fluid) / 1000
            ) / h_2_djuice_dec
        )
    )
### DECANTER AND FILTERING PROCESSES ### 
m_2_dec_juice = m_2_dosed_juice - m_2_flashvap #mass flow of the juice that enters the decanter
m_2_bagacilho = parameter_2[6] * m_2_juice #mass flow of bagacilho
m_2_w = parameter_2[5] * m_2_juice #mass flow of water for filtering dillution
m_2_filtercake = parameter_2[7] * m_2_juice #mass flow of filtercake
m_2_filter_juice = m_2_rec_juice * ( # mass flow of filtered juice that leaves the decanter and filtering processes
    1 - h_kadlec(brix_1_juice * 100, temperature(temperature_2[1], 'c'), parameter_1[1] * 100) / (cp('H', 'T', temperature_2[1], 'Q', 1, 'water') / 1000)
    ) / (1 - h_kadlec(brix_1_juice * 100, temperature(temperature_2[0], 'c'), parameter_1[1] * 100) / (cp('H', 'T', temperature_2[1], 'Q', 1, 'water') / 1000)
    ) 
m_2_steam_vaccum = m_2_filter_juice - m_2_rec_juice #steam that leaves the vaccum system
m_2_treated_juice = m_2_dec_juice + m_2_w + m_2_bagacilho - m_2_filtercake - m_2_filter_juice #mass flow of treat juice that leaves the subsystem 2
brix_2_treated = ( #brix of the treated juice
    m_2_juice * brix_1_juice * parameter_1[1] - m_2_filtercake * parameter_2[8] * parameter_2[9]
    ) / (m_2_treated_juice * parameter_1[1])
### VACCUM SYSTEM ###
m_2_w_invaccum = m_2_steam_vaccum * (
    cp('H', 'T', temperature_2[7], 'P', pressure_2[1], fluid) - cp('H', 'Q', 1, 'P', pressure_2[1], fluid)
    ) / (cp('H', 'T', temperature_2[6], 'P', pressure_2[1], fluid) - cp('H', 'T', temperature_2[7], 'P', pressure_2[1], fluid))
#mass_vaccum(m_2_steam_vaccum, temperature_2[6], temperature_2[7], parameter_2[11]) #mass of water needed in the vaccum system
m_2_w_outvaccum = m_2_steam_vaccum  + m_2_w_invaccum #mass of condensed water in the vaccum system
w_2 = [ #list that saves the work in the subsystem 2
    m_2_juice * (cp('H', 'T', temperature_2[3], 'P', pressure_2[1], fluid) - 
        cp('H', 'T', temperature_2[3], 'P', pressure_2[0], fluid)) / 1000, #[0] pump work of the inlet juice heating in kW
    m_2_filter_juice * h_kadlec( #[1] pump work of the vaccum system in kW
        brix_2_treated * 100, 
        temperature(temperature_2[1], 'c'), parameter_1[1] * 100
        ) + m_2_w_invaccum * cp(
            'H', 'P', p0, 'T', temperature_2[6], fluid
            ) / 1000 -(
                m_2_rec_juice * h_kadlec(
                    brix_2_treated * 100, 
                    temperature(temperature_2[0], 'c'), 
                    parameter_1[1] * 100
                    ) + m_2_w_outvaccum * cp(
                        'H', 'P', p0, 'T', temperature_2[7], fluid
                        ) / 1000
                        )
]
y_2_sucrose = mass_to_molar_frac(brix_2_treated, prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass']) #molar fraction of sucrose in raw juice
y_2_water = 1 - y_2_sucrose #molar frac

m_ex_2 = [
    - parameter_2[0] * m_ex_1[2], #[0] exergy from the extracted juice from subsystem 1 that enters the subsystem 2
    m_2_SO2 * prop_cp['sulfur_dioxide']['chem_x'], #[1] sulfur for sulfitation
    m_2_wCaO * (ph_exergy( #[2] water to dilute the CaO
        cp('H', 'T', temperature_2[9],'P', pressure_2[0], fluid) / 1000,
        cp('S', 'T', temperature_2[9],'P', pressure_2[0], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000,
    ) + prop_cp['water']['chem_x']),
    m_2_CaO * prop_cp['calcium_oxide']['chem_x'], # [3] CaO
    m_2_wh * (ph_exergy( #[4]
        h_2_whot,
        cp('S', 'Q', 1, 'P', pressure_2[2], fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    -m_2_wh * (ph_exergy( #[5]
        h_2_wcold,
        cp('S', 'Q', 0, 'P', pressure_2[2], fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    -m_2_flashvap * (ph_exergy( #[6]
        cp('H', 'T', temperature_2[8], 'Q', 1, fluid) / 1000,
        cp('S', 'T', temperature_2[8], 'Q', 1, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    - m_2_treated_juice * (mix_exergy( #[7] outflow juice
        activity_sucrose(
            y_2_sucrose,
            x_sol_peakcock(temperature(temperature_2[5], 'c')),
            temperature_2[5]
            ),
            activity_water(
                y_2_water,
                y_2_sucrose,
                temperature_2[5]
                ),
                prop_cp['sucrose']['molar_mass'] * y_2_sucrose + prop_cp['water']['molar_mass'] * y_2_water,
                y_2_sucrose, 
                y_2_water,
                t0
                )+ph_exergy(
                    h_kadlec(brix_2_treated * 100, temperature(temperature_2[5], 'c'), parameter_1[1] * 100), 
                    delta_s(brix_2_treated * 100, parameter_1[1] * 100, temperature_2[5], t0), 
                    h_kadlec(brix_2_treated * 100, temperature(t0, 'c'), parameter_1[1] * 100), 
                    0
                    ) + prop_cp['water']['chem_x'] * (1 - brix_2_treated) + prop_cp['sucrose']['chem_x'] * brix_2_treated
    ),
    - m_2_bagacilho / m_bagasse * m_ex_1[1], #[8]
    m_2_w * (ph_exergy( #[9]
        cp('H', 'T', temperature_2[9], 'P', pressure_2[0], fluid) / 1000,
        cp('S', 'T', temperature_2[9], 'P', pressure_2[0], fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    -m_2_w_outvaccum * (ph_exergy( #[10]
        cp('H', 'T', temperature_2[7], 'P', p0, fluid) / 1000,
        cp('S', 'T', temperature_2[7], 'P', p0, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    m_2_w_invaccum * (ph_exergy( #[11]
        cp('H', 'T', temperature_2[6], 'P', p0, fluid) / 1000,
        cp('S', 'T', temperature_2[6], 'P', p0, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x'])
]

print('Subsystem 2 - JUICE TREATMENT TO THE SUGAR PRODUCTION - exergy destruction: {:.2f} MW'.format(exergy_destruction(0, w_2, m_ex_2) / 1000))      
#####################################################################################################
############################## JUICE TREATMENT TO THE ETHANOL PRODUCTION (3) ########################
#####################################################################################################
m_3_juice = (1 - parameter_2[0]) * m_rawjuice #mass flow of the treated juice in kg/s
m_3_rec_juice = parameter_2[10] * m_3_juice #mass flow of the recirculated juice in kg/s
m_3_CaO = parameter_2[3] * m_3_juice #mass flow of CaO in kg/s
m_3_wCaO = parameter_2[4] * m_3_juice #mass flow of water for adding up to the CaO form lime in kg/s
m_3_dosed_juice = m_3_juice + m_3_CaO + m_3_wCaO + m_3_rec_juice #juice mass flow that was 
#chemical treated in kg/s

### HEATING PROCESS IS IMPORTED FROM SUBSYSTEM 2###
m_3_wh = m_3_dosed_juice * (h_2_djuice105 - h_2_djuice35) / (h_2_whot - h_2_wcold) #mass flow of water to heat the juice up after the liming process

### FLASH TANK (ISOTHERMIC EXPANSION) ###
m_3_flashvap = m_3_dosed_juice * ( #saturated steam flashed in the flash tank
    (
        1 - h_2_djuice105 / h_2_djuice_dec
    ) / (1 - (
        cp('H', 'T', temperature_2[8], 'Q', 1, fluid) / 1000
            ) / h_2_djuice_dec
        )
    )
### DECANTER AND FILTERING PROCESSES ### 
m_3_dec_juice = m_3_dosed_juice - m_3_flashvap #mass flow of the juice that enters the decanter
m_3_bagacilho = parameter_2[6] * m_3_juice #mass flow of bagacilho
m_3_w = parameter_2[5] * m_3_juice #mass flow of water for filtering dillution
m_3_filtercake = parameter_2[7] * m_3_juice #mass flow of filtercake
m_3_filter_juice = m_3_rec_juice * ( # mass flow of filtered juice that leaves the decanter and filtering processes
    1 - h_kadlec(brix_1_juice * 100, temperature(temperature_2[1], 'c'), parameter_1[1] * 100) / (cp('H', 'T', temperature_2[1], 'Q', 1, 'water') / 1000)
    ) / (1 - h_kadlec(brix_1_juice * 100, temperature(temperature_2[0], 'c'), parameter_1[1] * 100) / (cp('H', 'T', temperature_2[1], 'Q', 1, 'water') / 1000)
    ) 
m_3_steam_vaccum = m_3_filter_juice - m_3_rec_juice #steam that leaves the vaccum system
m_3_treated_juice = m_3_dec_juice + m_3_w + m_3_bagacilho - m_3_filtercake - m_3_filter_juice #mass flow of treat juice that leaves the subsystem 2
brix_3_treated = ( #brix of the treated juice
    m_3_juice * brix_1_juice * parameter_1[1] - m_3_filtercake * parameter_2[8] * parameter_2[9]
    ) / (m_3_treated_juice * parameter_1[1])

### VACCUM SYSTEM ###
m_3_w_invaccum = m_3_steam_vaccum * ( #mass of water needed in the vaccum system from P. Rein, 
#Cane sugar engineering, Second edition. Berlin: Verlag Dr. Albert Bartens KG, 2017. (page 321)
    570 / (temperature_2[7] - temperature_2[6])
    ) * (1 + parameter_2[11])
m_3_w_outvaccum = m_3_steam_vaccum  + m_3_w_invaccum #mass of condensed water in the vaccum system
w_3 = [ #list that saves the work in the subsystem 3
    m_3_juice * (cp('H', 'T', temperature_2[3], 'P', pressure_2[1], fluid) - 
        cp('H', 'T', temperature_2[3], 'P', pressure_2[0], fluid)) / 1000, #[0] pump work of the inlet juice heating in kW
    m_3_filter_juice * h_kadlec( #[1] pump work of the vaccum system in kW
        brix_3_treated * 100, 
        temperature(temperature_2[1], 'c'), parameter_1[1] * 100
        ) + m_3_w_invaccum * cp(
            'H', 'P', p0, 'T', temperature_2[6], fluid
            ) / 1000 -(
                m_3_rec_juice * h_kadlec(
                    brix_3_treated * 100, 
                    temperature(temperature_2[0], 'c'), 
                    parameter_1[1] * 100
                    ) + m_3_w_outvaccum * cp(
                        'H', 'P', p0, 'T', temperature_2[7], fluid
                        ) / 1000
                        )
]
y_3_sucrose = mass_to_molar_frac(brix_3_treated, prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass']) #molar fraction of sucrose in raw juice
y_3_water = 1 - y_3_sucrose #molar frac
m_ex_3 = [
    - (1 - parameter_2[0]) * m_ex_1[2], #[0] exergy from the extracted juice from subsystem 1 that enters the subsystem 2
    m_3_wCaO * (ph_exergy( #[1] water to dilute the CaO
        cp('H', 'T', temperature_2[9],'P', pressure_2[0], fluid) / 1000,
        cp('S', 'T', temperature_2[9],'P', pressure_2[0], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000,
    ) + prop_cp['water']['chem_x']),
    m_3_CaO * prop_cp['calcium_oxide']['chem_x'], #[2]
    m_3_wh * (ph_exergy( #[3]
        h_2_whot,
        cp('S', 'Q', 1, 'P', pressure_2[2], fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    -m_3_wh * (ph_exergy( #[4]
        h_2_wcold,
        cp('S', 'Q', 0, 'P', pressure_2[2], fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    -m_3_flashvap * (ph_exergy( #[5]
        cp('H', 'T', temperature_2[8], 'Q', 1, fluid) / 1000,
        cp('S', 'T', temperature_2[8], 'Q', 1, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    - m_3_treated_juice * (mix_exergy( #[6] outflow juice
        activity_sucrose(
            y_3_sucrose,
            x_sol_peakcock(temperature(temperature_2[5], 'c')),
            temperature_2[5]
            ),
            activity_water(
                y_3_water,
                y_3_sucrose,
                temperature_2[5]
                ),
                prop_cp['sucrose']['molar_mass'] * y_3_sucrose + prop_cp['water']['molar_mass'] * y_3_water,
                y_3_sucrose, 
                y_3_water,
                t0
                )+ph_exergy(
                    h_kadlec(brix_3_treated * 100, temperature(temperature_2[5], 'c'), parameter_1[1] * 100), 
                    delta_s(brix_3_treated * 100, parameter_1[1] * 100, temperature_2[5], t0), 
                    h_kadlec(brix_3_treated * 100, temperature(t0, 'c'), parameter_1[1] * 100), 
                    0
                    ) + prop_cp['water']['chem_x'] * (1 - brix_3_treated) + prop_cp['sucrose']['chem_x'] * brix_2_treated
    ),
    - m_3_bagacilho / m_bagasse * m_ex_1[1], #[7]
    m_3_w * (ph_exergy( #[8]
        cp('H', 'T', temperature_2[9], 'P', pressure_2[0], fluid) / 1000,
        cp('S', 'T', temperature_2[9], 'P', pressure_2[0], fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    -m_3_w_outvaccum * (ph_exergy( #[9]
        cp('H', 'T', temperature_2[7], 'P', p0, fluid) / 1000,
        cp('S', 'T', temperature_2[7], 'P', p0, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x']),
    m_3_w_invaccum * (ph_exergy( #[10]
        cp('H', 'T', temperature_2[6], 'P', p0, fluid) / 1000,
        cp('S', 'T', temperature_2[6], 'P', p0, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000,
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
    ) + prop_cp['water']['chem_x'])
]

print('Subsystem 3 - JUICE TREATMENT TO THE ETHANOL PRODUCTION - exergy destruction: {:.2f} MW'.format(exergy_destruction(0, w_3, m_ex_3) / 1000)) 
#####################################################################################################
###################################### JUICE EVAPORATION (4) ########################################
#####################################################################################################
#neglected sugar losses in the subsystem
parameter_4 = [
    0.65, #[0] outflow brix
    0.86, #[1] molasses purity
    0.86, #[2] clarified juice purity
    0.2   #[3] security factor for the vaccumm system
]
temperature_4 = [
    temperature(97, 'k'), #[0] temperature of the clarified juice
    temperature(30, 'k'), #[1] temperature of the inlet water that goes to the vaccumm system
    temperature(50, 'k'), #[2] temperature of the outlet water that leaves the vaccumm system
    0, #[3] 1st evaporator in C
    0, #[4] 2nd evaporator in C
    0, #[5] 3rd evaporator in C
    0, #[6] 4th evaporator in C
    0  #[7] 5th evaporator in C
]
pressure_4 = [
    pressure(1.69, 'pa'), #[0] pressure of the 1st effect
    pressure(1.31, 'pa'), #[1] pressure of the 2nd effect
    pressure(0.93, 'pa'), #[2] pressure of the 3rd effect
    pressure(0.54, 'pa'), #[3] pressure of the 4th effect
    pressure(0.16, 'pa'), #[4] pressure of the 5th effect
    pressure(2.5, 'pa'),  #[5] pressure of the turbine stage
    pressure(1, 'pa')     #[6] pressure of the vaccum system condensate
]
msugar = brix_2_treated * m_2_treated_juice #content of sugar in the subsystem in kg/s
m_4_evap = msugar / parameter_4[0] #mass flow of the evaporated juice in kg/s

def mf4(p, x):
    F = np.empty(10)
    mv0, ml1, mv1, ml2, mv2, ml3, mv3, ml4, mv4, mv5 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    ###P1
    brix1 = msugar / ml1
    t = temperature(cp('T', 'Q', 1, 'P', pressure_4[0], fluid), 'c')
    temperature_4[3] = t + delta_t_peackok(t, brix1 * 100)
    
    F[0] = ml1 + mv1 - m_2_treated_juice
    F[1] = mv0 * (
        cp('H', 'Q', 1, 'P', pressure_4[5], fluid) - cp('H', 'Q', 0, 'P', pressure_4[5], fluid)
        ) / 1000 + m_2_treated_juice * h_kadlec(brix_2_treated * 100, temperature(temperature_2[5], 'c'), parameter_1[1] * 100) - ml1 * h_kadlec(brix1 * 100, temperature_4[3], parameter_1[1] * 100) - x * mv1 * (cp('H', 'T', temperature(temperature_4[3], 'k'), 'P', pressure_4[0], fluid) / 1000)
    ###P2
    brix2 = msugar / ml2
    t = temperature(cp('T', 'Q', 1, 'P', pressure_4[1], fluid), 'c')
    temperature_4[4] = t + delta_t_peackok(t, brix2 * 100)

    F[2] = ml2 + mv2 - ml1
    F[3] = ml1 * h_kadlec(
        brix1 * 100, temperature_4[3], parameter_1[1] * 100
        ) + x * mv1 * (cp('H', 'T', temperature(temperature_4[3], 'k'), 'P', pressure_4[0], fluid) / 1000 - cp('H', 'Q', 0, 'P', pressure_4[1], fluid) / 1000) - mv2 * (cp('H', 'T', temperature(temperature_4[4], 'k'), 'P', pressure_4[1], fluid) / 1000) - ml2 * h_kadlec(brix2 * 100, temperature_4[4], parameter_1[1] * 100)
    ###P3
    brix3 = msugar / ml3
    t = temperature(cp('T', 'Q', 1, 'P', pressure_4[2], fluid), 'c')
    temperature_4[5] = t + delta_t_peackok(t, brix3 * 100)

    F[4] = ml3 + mv3 - ml2
    F[5] = ml2 * h_kadlec(
        brix2 * 100, temperature_4[4], parameter_1[1] * 100
        ) + mv2 * (cp('H', 'T', temperature(temperature_4[4], 'k'), 'P', pressure_4[1], fluid) / 1000 - cp('H', 'Q', 0, 'P', pressure_4[2], fluid) / 1000) - mv3 * (cp('H', 'T', temperature(temperature_4[5], 'k'), 'P', pressure_4[2], fluid) / 1000) - ml3 * h_kadlec(brix3 * 100, temperature_4[5], parameter_1[1] * 100)
    ###P4
    brix4 = msugar / ml4
    t = temperature(cp('T', 'Q', 1, 'P', pressure_4[3], fluid), 'c')
    temperature_4[6] = t + delta_t_peackok(t, brix4 * 100)

    F[6] = ml4 + mv4 - ml3
    F[7] = ml3 * h_kadlec(
        brix3 * 100, temperature_4[5], parameter_1[1] * 100
        ) + mv3 * (cp('H', 'T', temperature(temperature_4[5], 'k'), 'P', pressure_4[2], fluid) / 1000 - cp('H', 'Q', 0, 'P', pressure_4[3], fluid) / 1000) - mv4 * (cp('H', 'T', temperature(temperature_4[6], 'k'), 'P', pressure_4[3], fluid) / 1000) - ml4 * h_kadlec(brix4 * 100, temperature_4[6], parameter_1[1] * 100)
    ###P5
    t = temperature(cp('T', 'Q', 1, 'P', pressure_4[4], fluid), 'c')
    temperature_4[7] = t + delta_t_peackok(t, parameter_4[0] * 100)

    F[8] = m_4_evap + mv5 - ml4
    F[9] = ml4 * h_kadlec(
        brix4 * 100, temperature_4[6], parameter_1[1] * 100
        ) + mv4 * (cp('H', 'T', temperature(temperature_4[6], 'k'), 'P', pressure_4[3], fluid) / 1000 - cp('H', 'Q', 0, 'P', pressure_4[4], fluid) / 1000) - mv5 * (cp('H', 'T', temperature(temperature_4[7], 'k'), 'P', pressure_4[4], fluid) / 1000) - m_4_evap * h_kadlec(parameter_4[0] * 100, temperature_4[7], parameter_1[1] * 100)
    
    return F
stp = int((m_2_treated_juice - m_4_evap) / 10) #increment for the initial guest
initial_guest = np.array(list(range(int(m_2_treated_juice - stp), int(m_4_evap) + stp, -stp))) # initial guest
massflow_4 = fsolve(mf4, initial_guest, 1)
#massflow_4[0] = mv0, massflow_4[1] = ml1, massflow_4[2] = mv1, massflow_4[3] = ml2, massflow_4[4] = mv2,
#massflow_4[5] = ml3, massflow_4[6] = mv3, massflow_4[7] = ml4, massflow_4[8] = mv4, massflow_4[9] = mv5
#
#                          mv1______   mv2______   mv3______   mv4______   mv5---->
#                          |        |  |        |  |        |  |        |  |
#                       ___|_       |__|_       |__|_       |__|_       |__|_
#                 mv0--> \(1)|     | \(2)|     | \(3)|     | \(4)|     | \(5)|
#                      |  \  |     |  \  |     |  \  |     |  \  |     |  \  |
#                      |   \ |     |   \ |     |   \ |     |   \ |     |   \ |
#   m_2_treated_juice-->____\|-ml1->____\|-ml2->____\|-ml3->____\|-ml4->____\|-m_4_evap->
#                           |           |           |           |           |
#                           mv0         mv1         mv2         mv3         mv4
brix_4 = [
    msugar / massflow_4[1], #[0] brix of the 1st evaporator
    msugar / massflow_4[3], #[1] brix of the 2st evaporator
    msugar / massflow_4[5], #[2] brix of the 3rd evaporator
    msugar / massflow_4[7]  #[3] brix of the 4th evaporator
    #brix of the 5th evaporator -> parameter_4[0]
]
###VACCUM SYSTEM###
m_4_w_invaccum = massflow_4[9] * (
     cp('H', 'Q', 1, 'P', pressure_4[4], fluid) - cp('H', 'T', temperature_4[2], 'P', pressure_4[6], fluid)
    ) / (cp('H', 'T', temperature_4[2], 'P', pressure_4[6], fluid) - cp('H', 'T', temperature_4[1], 'P', pressure_4[6], fluid))
m_4_w_outvaccum = massflow_4[9] + m_4_w_invaccum

y_4_sucrose = mass_to_molar_frac(parameter_4[0], prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass']) #molar fraction of sucrose in raw juice
y_4_water = 1 - y_2_sucrose #molar frac
m_ex_4 = [
    - m_ex_2[7], #[0] inflow juice
    massflow_4[0] * (ph_exergy( #[1]
        cp('H', 'Q', 1, 'P', pressure_4[5], fluid) / 1000,
        cp('S', 'Q', 1, 'P', pressure_4[5], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    m_4_w_invaccum * (ph_exergy( #[2]
        cp('H', 'T', temperature_4[1], 'P', pressure_4[6], fluid) / 1000,
        cp('S', 'T', temperature_4[1], 'P', pressure_4[6], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - m_4_evap * (mix_exergy( #[3] syrup
        activity_sucrose(
            y_4_sucrose,
            x_sol_peakcock(temperature_4[7]),
            temperature(temperature_4[7], 'k')
            ),
            activity_water(
                y_4_water,
                y_4_sucrose,
                temperature(temperature_4[7], 'k')
                ),
                prop_cp['sucrose']['molar_mass'] * y_4_sucrose + prop_cp['water']['molar_mass'] * y_4_water,
                y_4_sucrose, 
                y_4_water,
                t0
                )+ph_exergy(
                    h_kadlec(parameter_4[0] * 100, temperature_4[7], parameter_1[1] * 100), 
                    delta_s(parameter_4[0] * 100, parameter_1[1] * 100, temperature(temperature_4[7], 'k'), t0), 
                    h_kadlec(parameter_4[0] * 100, temperature(t0, 'c'), parameter_1[1] * 100), 
                    0
                    ) + prop_cp['water']['chem_x'] * (1 - parameter_4[0]) + prop_cp['sucrose']['chem_x'] * parameter_4[0]
    ),
    - m_4_w_outvaccum * (ph_exergy( #[4]
        cp('H', 'T', temperature_4[2], 'P', pressure_4[6], fluid) / 1000,
        cp('S', 'T', temperature_4[2], 'P', pressure_4[6], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']), #massflow_4[8] = mv4, massflow_4[9] = mv5
    - massflow_4[0] * (ph_exergy( #[5]
        cp('H', 'Q', 0, 'P', pressure_4[5], fluid) / 1000,
        cp('S', 'Q', 0, 'P', pressure_4[5], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - massflow_4[2] * (ph_exergy( #[6]
        cp('H', 'Q', 0, 'P', pressure_4[0], fluid) / 1000,
        cp('S', 'Q', 0, 'P', pressure_4[0], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - massflow_4[4] * (ph_exergy( #[7]
        cp('H', 'Q', 0, 'P', pressure_4[1], fluid) / 1000,
        cp('S', 'Q', 0, 'P', pressure_4[1], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - massflow_4[6] * (ph_exergy( #[8]
        cp('H', 'Q', 0, 'P', pressure_4[2], fluid) / 1000,
        cp('S', 'Q', 0, 'P', pressure_4[2], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - massflow_4[8] * (ph_exergy( #[9]
        cp('H', 'Q', 0, 'P', pressure_4[3], fluid) / 1000,
        cp('S', 'Q', 0, 'P', pressure_4[3], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x'])
]

print('Subsystem 4 - EVAPORATION - exergy destruction: {:.2f} MW'.format(exergy_destruction(0, 0, m_ex_4) / 1000))
#####################################################################################################
###################################### SUGAR CRISTALIZATION (5) #####################################
#####################################################################################################
parameter_5 = [
    0.78,  #[0] sugar production (kg/kg of syrup)
    2,     #[1] specific energy to power the centrifuges in kJ/kg of massecuite
    0.02,  #[2] water to wash the centrifuges in kg_H2O/kg of massecuite
    0.0415,#[3] water to the B cooker in kg_h2O/kg of A sugar
    0.8932,#[4] fraction of syrup that goes to the sugar production
]
temperature_5 = [
    60, #[0] temperature of the rich syrup in C
    57, #[1] temperature of the poor syrup in C
    57, #[2] temperature of the B sugar in C
    57, #[3] temperature of the molasses in C
    25, #[4] temperature of the A sugar in C
    30, #[5] temperature of the inlet vaccum water in C
    50, #[6] temperature of the outlet vaccum water in C
    0,  #[7] temperature of the A COOKER in C
    0   #[8] temperature of the B COOKER in C
]
pressure_5 = [
    pressure(6, 'pa'),   #[0] pressure of the water line to clean the centrifuges and to dilute sugar
    pressure(1.69, 'pa'),#[1] pressure of the steam line in Pa
    pressure(0.16, 'pa'),#[2] pressure of the pans in Pa
    pressure(1.013, 'pa')#[3] pressure of the vaccum system waterline
]
brix_5 = [
    0.94, #[0] brix of the A massecuite
    0.73, #[1] brix of the rich syrup
    0.82, #[2] brix of the poor syrup
    0.79, #[3] brix of the syrup that goes to the B cooker
    0.93, #[4] brix of the B massecuite
    0.999,#[5] brix of the B sugar
    0.73, #[6] brix of the molasses
    0,    #[7] brix of the diluited B sugar
    0.999,#[8] brix of the A sugar
    0     #[9] brix of the recirculated B sugar
]
pz_5 = [
    0.81, #[0] purity of the A massecuite
    0.69, #[1] purity of the rich syrup
    0.65, #[2] purity of the poor syrup
    0.65, #[3] purity of the syrup that goes to the B cooker
    0,    #[4] purity of the B massecuite
    0.88, #[5] purity of the B sugar
    0.51, #[6] purity of the molasses
    0,    #[7] purity of the diluited B sugar
    0.997,#[8] purity of the A sugar
    0     #[9] purity of the recirculated B sugar
]
m_5_evap = m_4_evap * parameter_5[4]
sugar = m_5_evap * parameter_4[0] * parameter_4[2]
m_5_sugar = parameter_5[0] * sugar
m_5_syrup = sugar - m_5_sugar
m_5_water = [
    0,
    0
]
def mf5(p):
    F = np.empty(11)
    me2, me3, me4, me5, me7, me8, me9, me13, me15, te11, te114 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]
    ###COOKER A
    t = temperature(cp('T', 'Q', 1, 'P', pressure_5[2], fluid), 'c') 
    temperature_5[7] = t + delta_t_peackok(t, brix_5[0] * 100)
    me14 = me13 - m_5_syrup
    me114 = me14 + me15
    F[0] = m_5_evap + me114 + me8 - me4 - me7
    brix_5[7] = me14 * brix_5[5] / me114
    pz_5[7] = me14 * brix_5[5] * pz_5[5] / (me114 * brix_5[7])
    F[1] = m_5_evap * h_kadlec(parameter_4[0] * 100, temperature_4[7], parameter_1[1] * 100) + me114 * h_kadlec(brix_5[7] * 100, te114, pz_5[7] * 100) + me8 * h_kadlec(brix_5[1] * 100, temperature_5[0], pz_5[1] * 100) + me2 * (cp('H', 'Q', 1, 'P', pressure_5[1], fluid) - cp('H', 'Q', 0, 'P', pressure_5[1], fluid)) / 1000 - me4 * cp('H', 'T', temperature(temperature_5[7], 'k'), 'P', pressure_5[2], fluid) / 1000 - me7 * h_kadlec(brix_5[0] * 100, temperature_5[7], pz_5[0] * 100)
    ###A CENTRIFUGE
    me10 = me7 * parameter_5[2]
    F[2] = me7 + me10 - me8 - me9 - m_5_sugar
    F[3] = me7 * brix_5[0] * pz_5[0] - me8 * brix_5[1] * pz_5[1] - me9 * brix_5[2] * pz_5[2] - m_5_sugar * brix_5[8] * pz_5[8]
    F[4] = me7 * h_kadlec(brix_5[0] * 100, temperature_5[7], pz_5[0] * 100) + me7 * parameter_5[1] + me10 * cp('H', 'Q', 0, 'P', pressure_5[0], fluid) / 1000 - me8 * h_kadlec(brix_5[1] * 100, temperature_5[0], pz_5[1] * 100) - me9 * h_kadlec(brix_5[2] * 100, temperature_5[1], pz_5[2] * 100) - m_5_sugar * h_kadlec(brix_5[8] * 100, temperature_5[4], pz_5[8] * 100)
    ###DILUITION OF POOR SYRUP
    me11 = me9 * brix_5[2] * pz_5[2] / (brix_5[3] * pz_5[3])
    me12 = me11 - me9
    #F[5] = me9 * brix_5[2] * pz_5[2] - me11 * brix_5[3] * pz_5[3]
    F[5] = me9 * h_kadlec(brix_5[2] * 100, temperature_5[1], pz_5[2] * 100) + me12 * cp('H', 'Q', 0, 'P', pressure_5[0], fluid) / 1000 - me11 * h_kadlec(brix_5[3] * 100, temperature_5[2], pz_5[3] * 100)
    ###COOKER B
    temperature_5[8] = t + delta_t_peackok(t, brix_5[4] * 100)
    me18 = parameter_5[3] * sugar
    pz_5[4] = me11 * brix_5[3] * pz_5[3] / (me13 * brix_5[4])
    F[6] = me11 + me18 - me13 - me5
    F[7] = me11 * h_kadlec(brix_5[3] * 100, te11, pz_5[3] * 100) + me3 * (cp('H', 'Q', 1, 'P', pressure_5[1], fluid) - cp('H', 'Q', 0, 'P', pressure_5[1], fluid)) / 1000 + me18 * cp('H', 'Q', 0, 'P', pressure_5[0], fluid) / 1000 - me13 * h_kadlec(brix_5[4] * 100, temperature_5[8], pz_5[4] * 100) - me5 * cp('H', 'T', temperature(temperature_5[8], 'k'), 'P', pressure_5[2], fluid) / 1000
    ###B CENTRIFUGE
    F[8] = me13 * brix_5[4] * pz_5[4] - me14 * brix_5[5] * pz_5[5] - m_5_syrup * brix_5[6] * pz_5[6]
    F[9] = me13 * h_kadlec(brix_5[4] * 100, temperature_5[8], pz_5[4] * 100) + me13 * parameter_5[1] - me14 * h_kadlec(brix_5[5] * 100, temperature_5[2], pz_5[5] * 100) - m_5_syrup * h_kadlec(brix_5[6] * 100, temperature_5[3], pz_5[6] * 100)
    ###DILUITION OF B SUGAR
    brix_5[9] = me14 * brix_5[5] / me114
    pz_5[9] = me14 * brix_5[5] * pz_5[5] / (me114 * brix_5[9])
    F[10] = me14 * h_kadlec(brix_5[5] * 100, temperature_5[2], pz_5[5] * 100) + me15 * cp('H', 'Q', 0, 'P', pressure_5[0], fluid) / 1000 - me114 * h_kadlec(brix_5[9] * 100, te114, pz_5[9] * 100)
    m_5_water[0] = me10 + me12 + me18
    m_5_water[1] = (
        me4 * cp(
            'H', 'T', temperature(temperature_5[7], 'k'), 'P', pressure_5[2], fluid
            ) / 1000 + me5 * cp(
                'H', 'T', temperature(temperature_5[8], 'k'), 'P', pressure_5[2], fluid) / 1000) / (
                    cp(
                        'H', 'T', temperature(temperature_5[6], 'k'), 'P', pressure_5[3], fluid
                        ) / 1000 - cp(
                            'H', 'T', temperature(temperature_5[5], 'k'), 'P', pressure_5[3], fluid
                            ) / 1000
                )
    return F

alpha = m_5_evap * parameter_5[4] / 20.9
initial_guest = [
    11.8 * alpha, #[0] me2
    2.2 * alpha ,  #[1] me3
    8.3 * alpha ,  #[2] me4
    1.5 * alpha ,  #[3] me5
    22.7 * alpha , #[4] me7
    8.8 * alpha ,  #[5] me8
    7.1 * alpha ,  #[6] me9
    #0.3 * alpha,  #[7] me12
    6.2 * alpha ,  #[8] me13
    1.2 * alpha ,  #[9] me15
    45.3, #[13] te11
    50    #[14] te114
]
massflow_5 = fsolve(mf5, initial_guest)
w_5 = [
    -massflow_5[4] * parameter_5[1],
    -massflow_5[8] * parameter_5[1]
]
y_5_sucrose = [
    y_4_sucrose, #[0]
    mass_to_molar_frac(brix_5[6], prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass']), #molar fraction of sucrose in raw juice
    mass_to_molar_frac(brix_5[8], prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass']) #molar fraction of sucrose in raw juice
]
y_5_water =  list(map(lambda x: 1 - x, y_5_sucrose))
m_ex_5 = [
    m_5_evap * (mix_exergy( #[0] inflow syrup   
        activity_sucrose(
            y_5_sucrose[0],
            x_sol_peakcock(temperature_4[7]),
            temperature(temperature_4[7], 'k')
            ),
            activity_water(
                y_5_water[0],
                y_5_sucrose[0],
                temperature(temperature_4[7], 'k')
                ),
                prop_cp['sucrose']['molar_mass'] * y_5_sucrose[0] + prop_cp['water']['molar_mass'] * y_5_water[0],
                y_5_sucrose[0], 
                y_5_water[0],
                t0
                )+ph_exergy(
                    h_kadlec(parameter_4[0] * 100, temperature_4[7], parameter_1[1] * 100), 
                    delta_s(parameter_4[0] * 100, parameter_1[1] * 100, temperature(temperature_4[7], 'k'), t0), 
                    h_kadlec(parameter_4[0] * 100, temperature(t0, 'c'), parameter_1[1] * 100), 
                    0
                    ) + prop_cp['water']['chem_x'] * (1 - parameter_4[0]) + prop_cp['sucrose']['chem_x'] * parameter_4[0]
        ),
    (massflow_5[0] + massflow_5[1]) * (ph_exergy( #[1]
        cp('H', 'Q', 1, 'P', pressure_5[1], fluid) / 1000,
        cp('S', 'Q', 1, 'P', pressure_5[1], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    m_5_water[1] * (ph_exergy( #[2]
        cp('H', 'T', temperature(temperature_5[5], 'k'), 'P', pressure_5[3], fluid) / 1000,
        cp('S', 'T', temperature(temperature_5[5], 'k'), 'P', pressure_5[3], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - m_5_sugar * (mix_exergy( #[3] inflow syrup   
        activity_sucrose(
            y_5_sucrose[1],
            x_sol_peakcock(temperature_5[4]),
            temperature(temperature_5[4], 'k')
            ),
            activity_water(
                y_5_water[1],
                y_5_sucrose[1],
                temperature(temperature_5[4], 'k')
                ),
                prop_cp['sucrose']['molar_mass'] * y_5_sucrose[1] + prop_cp['water']['molar_mass'] * y_5_water[1],
                y_5_sucrose[1], 
                y_5_water[1],
                t0
                )+ph_exergy(
                    h_kadlec(brix_5[8] * 100, temperature_5[4], pz_5[8] * 100), 
                    delta_s(brix_5[8] * 100, pz_5[8] * 100, temperature(temperature_5[4], 'k'), t0), 
                    h_kadlec(brix_5[8] * 100, temperature(t0, 'c'), pz_5[8] * 100), 
                    0
                    ) + prop_cp['water']['chem_x'] * (1 - brix_5[8]) + prop_cp['sucrose']['chem_x'] * brix_5[8]
        ),
    - m_5_syrup * (mix_exergy( #[4] outflow syrup   
        activity_sucrose(
            y_5_sucrose[2],
            x_sol_peakcock(temperature_5[3]),
            temperature(temperature_5[3], 'k')
            ),
            activity_water(
                y_5_water[2],
                y_5_sucrose[2],
                temperature(temperature_5[3], 'k')
                ),
                prop_cp['sucrose']['molar_mass'] * y_5_sucrose[2] + prop_cp['water']['molar_mass'] * y_5_water[2],
                y_5_sucrose[2], 
                y_5_water[2],
                t0
                )+ph_exergy(
                    h_kadlec(brix_5[6] * 100, temperature_5[3], pz_5[6] * 100), 
                    delta_s(brix_5[6] * 100, pz_5[6] * 100, temperature(temperature_5[3], 'k'), t0), 
                    h_kadlec(brix_5[6] * 100, temperature(t0, 'c'), pz_5[6] * 100), 
                    0
                    ) + prop_cp['water']['chem_x'] * (1 - brix_5[6]) + prop_cp['sucrose']['chem_x'] * brix_5[6]
        ),
    - (m_5_water[1] + massflow_5[2]) * (ph_exergy( #[5] 
        cp('H', 'T', temperature(temperature_5[6], 'k'), 'P', pressure_5[3], fluid) / 1000,
        cp('S', 'T', temperature(temperature_5[6], 'k'), 'P', pressure_5[3], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - massflow_5[0] * (ph_exergy( #[6]
        cp('H', 'Q', 0, 'P', pressure_5[1], fluid) / 1000,
        cp('S', 'Q', 0, 'P', pressure_5[1], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - massflow_5[1] * (ph_exergy( #[7]
        cp('H', 'Q', 0, 'P', pressure_5[1], fluid) / 1000,
        cp('S', 'Q', 0, 'P', pressure_5[1], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x'])
]

print('Subsystem 5 - SUGAR CRYSTALIZATION - exergy destruction: {:.2f} MW'.format(exergy_destruction(0, w_5, m_ex_5) / 1000))
#####################################################################################################
###################################### SUGAR DRYING (6) #############################################
#####################################################################################################
parameter_6 = [
    0.019, #[0] air humidity at room temperature (kg of water / kg of dry air)
    0.036, #[1] air humidity that leaves the sugar dryer (kg of water / kg of dry air)
    0.0001,#[2] sugar humidity that leaves the sugar dryer (kg of water / kg of sugar)
]
temperature_6 = [
    100, #[0] temperature of the heated air
    25   #[1] temperature of the outlet sugar
]
pressure_6 = [
    pressure(2.5, 'pa') #pressure of the steam line
]
m_6_sugar = m_5_sugar * brix_5[8] + sugar * parameter_6[2]
brix_6 = m_5_sugar * brix_5[8] / m_6_sugar
pz_6 = brix_5[8] * pz_5[8] / brix_6
m_6_air = (m_5_sugar * (1 - brix_5[8]) - sugar * parameter_6[2]) / (parameter_6[1] - parameter_6[0])
m_6_uair = m_5_sugar + m_6_air - m_6_sugar
m_6_steam = m_6_air * (
    air_cp(
        'H', 'T', temperature(temperature_6[0], 'k'), 'P', p0, 'W', parameter_6[0]
        ) -  air_cp(
            'H', 'T', t0, 'P', p0, 'W', parameter_6[0]
            )
            ) / (
                cp('H', 'Q', 1, 'P', pressure_6[0], fluid) - cp('H', 'Q', 0, 'P', pressure_6[0], fluid)
            )


h_6_uair = (m_5_sugar * h_kadlec(
        brix_5[8] * 100, temperature_5[4], pz_5[8] * 100) + m_6_air * air_cp(
            'H', 'T', temperature(temperature_6[0], 'k'), 'P', p0, 'W', parameter_6[0]
            ) / 1000 - m_6_sugar * h_kadlec(
                brix_6 * 100, temperature_6[1], pz_6 * 100
                )
                ) / m_6_uair

y_6_sucrose = mass_to_molar_frac(brix_6, prop_cp['sucrose']['molar_mass'], prop_cp['water']['molar_mass']) #molar fraction of sucrose in raw juice
y_6_water = 1 - y_6_sucrose

m_ex_6 = [
    - m_ex_5[4], #[0]
    m_6_air * (ph_exergy( #[1]
        air_cp('H', 'T', t0, 'P', p0, 'W', parameter_6[0]) / 1000,
        air_cp('S', 'T', t0, 'P', p0, 'W', parameter_6[0]) / 1000,
        air_cp('H', 'T', t0, 'P', p0, 'W', parameter_6[0]) / 1000,
        air_cp('S', 'T', t0, 'P', p0, 'W', parameter_6[0]) / 1000,
        ) + prop_cp['air']['chem_x']),
    m_6_steam * (ph_exergy( #[2]
        cp('H', 'Q', 1, 'P', pressure_6[0], fluid) / 1000,
        cp('S', 'Q', 1, 'P', pressure_6[0], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - m_6_sugar * (mix_exergy( #[0] inflow syrup   
        activity_sucrose(
            y_6_sucrose,
            x_sol_peakcock(temperature_6[1]),
            temperature(temperature_6[1], 'k')
            ),
            activity_water(
                y_6_water,
                y_6_sucrose,
                temperature(temperature_6[1], 'k')
                ),
                prop_cp['sucrose']['molar_mass'] * y_6_sucrose + prop_cp['water']['molar_mass'] * y_6_water,
                y_6_sucrose, 
                y_6_water,
                t0
                )+ph_exergy(
                    h_kadlec(brix_6 * 100, temperature_6[1], pz_6 * 100), 
                    delta_s(brix_6 * 100, pz_6 * 100, temperature(temperature_6[1], 'k'), t0), 
                    h_kadlec(brix_6 * 100, temperature(t0, 'c'), pz_6 * 100), 
                    0
                    ) + prop_cp['water']['chem_x'] * (1 - brix_6) + prop_cp['sucrose']['chem_x'] * brix_6
        ),
    - m_6_uair * (ph_exergy( #[1]
        h_6_uair,
        air_cp('S', 'H', h_6_uair * 1000, 'P', p0, 'W', parameter_6[1]) / 1000,
        air_cp('H', 'T', t0, 'P', p0, 'W', parameter_6[1]) / 1000,
        air_cp('S', 'T', t0, 'P', p0, 'W', parameter_6[1]) / 1000,
        ) + prop_cp['air']['chem_x']),
    m_6_steam * (ph_exergy( #[2]
        cp('H', 'Q', 0, 'P', pressure_6[0], fluid) / 1000,
        cp('S', 'Q', 0, 'P', pressure_6[0], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x'])
]
print('Subsystem 6 - SUGAR DRYING - exergy destruction: {:.2f} MW'.format(exergy_destruction(0, 0, m_ex_6) / 1000))
#####################################################################################################
################################### JUICE FERMENTATION  (7) ##########################################
#####################################################################################################
parameter_7 = [
    0.068,      #[0] ºINPM - wine
    0.89,       #[1] fermentation efficiency
    0.125,      #[2] fraction in volume of yeast in the fermentated wine
    0.6,        #[3] fraction in volume of yeast in the centrifugated yeast milk
    0.3,        #[4] fraction in volume of yeast in the treated yeast milk
    0.005,      #[5] fraction of yeast in the centrifugated wine
    0.00000013, #[6] consumption of sulphuric acid (kg of H2SO4 / kg of produced ethanol)
    575.4,      #[7] enthalpy of the fermentation reaction (kJ / kg ART conv)
    2100,       #[8] density of yeast in kg/m3
    2,          #[9] specific energy to power the centrifuges in kJ/kg of massflow
]
temperature_7 = [
    32, #[0] temperature of the fermentation in C
    25, #[1] inlet temperature of water for cooling in C
    30  #[2] outlet temperature of water for cooling in C
]
pressure_7 = [
    pressure(6, 'pa') #[0] pressure of the cooling water
]
m_7_evap = m_4_evap * (1 - parameter_5[4])
m_7_mosto = m_5_syrup + m_3_treated_juice + m_7_evap
brix_7_mosto = (m_5_syrup * brix_5[6] + m_3_treated_juice * brix_3_treated + m_7_evap * parameter_4[0]) / m_7_mosto
pz_7_mosto = (m_5_syrup * brix_5[6] * pz_5[6] + m_3_treated_juice * brix_3_treated * parameter_1[1] + m_7_evap * parameter_4[0] * parameter_1[1]) / (m_7_mosto * brix_7_mosto)

h_7_mosto = (m_5_syrup * h_kadlec(brix_5[6] * 100, temperature_5[3], pz_5[6] * 100) + m_3_treated_juice * h_kadlec(brix_3_treated * 100, temperature(temperature_2[5], 'c'), parameter_1[1] * 100) + m_7_evap * h_kadlec(parameter_4[0] * 100, temperature_4[7], parameter_1[1] * 100)) / m_7_mosto
q_cool_7 = m_7_mosto * (h_kadlec(brix_7_mosto * 100, temperature_7[0], pz_7_mosto * 100) - h_7_mosto)

m_w_7 = q_cool_7 / ((cp('H', 'T', temperature(temperature_7[1], 'k'), 'P', pressure_7[0], fluid) - cp('H', 'T', temperature(temperature_7[2], 'k'), 'P', pressure_7[0], fluid)) / 1000)

m_7_sugar = m_7_mosto * brix_7_mosto * pz_7_mosto

def fermentation(sg, cf_sugar):
    p = [
        sg * cf_sugar * 4 * prop_cp['ethanol']['molar_mass'] / prop_cp['sucrose']['molar_mass'], #ethanol
        sg * cf_sugar * 4 * prop_cp['carbon_dioxide']['molar_mass'] / prop_cp['sucrose']['molar_mass'] #carbon dioxide
    ]
    return p
ferm = fermentation(m_7_sugar, parameter_7[1])
m_7_out = ferm[0] / parameter_7[0]
m_7_h2so4 = parameter_7[6] * ferm[0]
m_7_dw = ferm[1] + m_7_out - m_7_mosto - m_7_h2so4

def density_v(alpha_1, d1, d2):
    return alpha_1 * d1 + (1 - alpha_1) * d2
rho_5 = density_v(parameter_7[4], parameter_7[8], cp('D', 'T', t0, 'P', p0, fluid))
rho_4 = density_v(parameter_7[3], parameter_7[8], cp('D', 'T', t0, 'P', p0, fluid))
m_7_yeast = (m_7_dw + m_7_h2so4) * parameter_7[4] / (rho_5 * (parameter_7[3] / rho_4 - parameter_7[4] / rho_5))
m_7_eth = m_7_out + m_7_yeast

w_7 = [
    - parameter_7[9] * m_7_eth
]
m_w2_7 = (parameter_7[7] * m_7_sugar) / ((cp('H', 'T', temperature(temperature_7[2], 'k'), 'P', pressure_7[0], fluid) - cp('H', 'T', temperature(temperature_7[1], 'k'), 'P', pressure_7[0], fluid)) / 1000)
n_7_eth = m_7_out * parameter_7[0] / prop_cp['ethanol']['molar_mass']
n_7_water = m_7_out * (1 - parameter_7[0]) / prop_cp['water']['molar_mass']
n_7_total = n_7_eth + n_7_water
y_7_eth = n_7_eth / n_7_total
y_7_water = 1 - y_7_eth
heos_water_eth = 'HEOS::ETHANOL['+str(y_7_eth)+']&WATER['+str(y_7_water)+']' #string that describes the composition of water-ethanol mixture

#int(cp.dt) for heat exergy content -> 1st part
def int_first(f, c, s, i, t_in, t_out):
    #f is the polynomium coeficients
    #c is the order
    #s is the sum
    if i <= len(f)-1:
        if c > 0:
            s+=f[i]/c*(t_out**c-t_in**c)
            i+=1
            return int_first(f, c - 1, s, i, t_in, t_out)
        else:
            return s
    else:
        return s 
def int_second(f, c, s, i, t_in, t_out):
    if i<=len(f)-1:
        if c>0:
            s+=-t0*f[i]/c*(t_out**c-t_in**c)
            i+=1
            return int_second(f, c-1, s, i, t_in, t_out)
        else:
            return s-t0*f[i]*np.log(t_out/t_in)
    else:
        return s

t_7_q = list(range(int(temperature(temperature_7[1] - 10, 'k')), int(temperature(temperature_7[2] + 10, 'k'))))
cp_7_q = list(map(lambda x: cp('C', 'T', x, 'P', p0, fluid) / 1000, t_7_q))
cof_7 = list(np.polyfit(t_7_q,cp_7_q,1))
q_ex_7 = int_first(cof_7, len(cof_7), 0, 0, temperature(temperature_7[1], 'k'), temperature(temperature_7[2], 'k')) + int_second(cof_7, len(cof_7)-1, 0, 0, temperature(temperature_7[1], 'k'), temperature(temperature_7[2], 'k'))
q_7 = [
    - m_w_7 * q_ex_7, #mash preparation heat losing
    - m_w2_7 * q_ex_7 #dorn
]

def h_mix_eth(ye, T):
    a0 = - 363868 + 1838.29 * T - 2.32763 * T ** 2
    a05 = 925982 - 4835.86 *  T + 6.37228 * T ** 2
    a15 = - 1404894 + 7516.61 * T - 10.1128 * T ** 2
    a25 = 1091318 - 5894.98 * T + 7.98868 * T ** 2
    a45 = -279986 + 1505.57 * T - 2.03127 * T ** 2
    return (
        ye * (1 - ye) * (a0 * ye + a05 * (ye ** 0.5) + a15 * (ye ** 1.5) + a25 * (ye ** 2.5) + a45 * (ye ** 4.5))
        ) / (1000 * (ye * prop_cp['ethanol']['molar_mass'] + (1 - ye) * prop_cp['water']['molar_mass']))

def s_mix_eth(ye, ae, yw, aw, T):
    #ye = molar fraction of ethanol
    #ae = activity coeficient of ethanol
    #yw = molar fraction of water
    #aw = activity coeficient of water
    #T = temperature in K
    return (h_mix_eth(ye, T) - mix_exergy(ae, aw, prop_cp['ethanol']['molar_mass'] * ye + prop_cp['water']['molar_mass'] * yw, ye, yw, T)) / T

m_ex_7 = [
    - m_ex_3[6], #[0]
    (parameter_5[4] - 1) * m_ex_4[3], #[1]
    - m_ex_5[4], #[2] -> syrup
    m_7_h2so4 * prop_cp['sulfuric_acid']['chem_x'], #[3]
    m_7_dw * (ph_exergy( #[4]
        cp('H', 'T', temperature(temperature_7[1], 'k'),'P', p0, fluid) / 1000,
        cp('S', 'T', temperature(temperature_7[1], 'k'),'P', p0, fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - ferm[1] * (ph_exergy( #[4]
        cp('H', 'T', temperature(temperature_7[0], 'k'),'P', p0, 'CO2') / 1000,
        cp('S', 'T', temperature(temperature_7[0], 'k'),'P', p0, 'CO2') / 1000,
        cp('H', 'T', t0,'P', p0, 'CO2') / 1000,
        cp('S', 'T', t0,'P', p0, 'CO2') / 1000
        ) + prop_cp['carbon_dioxide']['chem_x']),
    - m_7_out * (mix_exergy( #[5]
                y_7_eth,
                y_7_water,
                prop_cp['ethanol']['molar_mass'] * y_7_eth + prop_cp['water']['molar_mass'] * y_7_water,
                y_7_eth, 
                y_7_water,
                t0
                )+ph_exergy(
                    h_mix_eth(y_7_eth, temperature(temperature_7[0], 'k')),
                    s_mix_eth(y_7_eth, y_7_eth, y_7_water, y_7_water, temperature(temperature_7[0], 'k')),
                    h_mix_eth(y_7_eth, t0),
                    s_mix_eth(y_7_eth, y_7_eth, y_7_water, y_7_water, t0),
                    ) + prop_cp['water']['chem_x'] * (1 - parameter_7[0]) + prop_cp['ethanol']['chem_x'] * parameter_7[0]
        )
]

print('Subsystem 7 - JUICE FERMENTATION - exergy destruction: {:.2f} MW'.format(exergy_destruction(q_7, w_7, m_ex_7) / 1000))
############################################################################################################################
############################################### WINE DISTILLATION (8) ###################################################
############################################################################################################################
parameter_8 = [
#ENSINAS, A. V. Integração termica e otimização termoeconomica aplicadas ao processo industrial de 
#produção de açucar e etanol a partir da cana-de-açucar. 29 mar. 2018. 
    0.912,  #[0] INPM degree of the second grade ethanol
    0.937,  #[1] INPM degree of hydrated ethanol
    0.0002, #[2] INPM degree of vinasse
    0.0002, #[3] INPM degree of flegmasse
    0.5,    #[4] INPM degree of flegma   
    99,     #[5] reflux ratio of the first column
    3.5,    #[6] reflux ratio of the second column
    14.56   #[7] constant to calculate the exergy of vinasse and effluent
]
temperature_8 = [
    60,   #[0] temperature of the wine after it leaves the first heat exchanger in ºC
    78.1, #[1] temperature of the columns' heads 
    100,  #[2] temperature of the vinasse when it leaves the first column in ºC
    100,  #[3] temperature of the flegmasse when it leaves the second column in ºC
    35,   #[4] temperature of the second grade and hydrated ethanol in ºC
    90,   #[5] temperature of the wine before it enters the first column in ºC
    30,   #[6] inlet temperature of the cooling water in ºC
    50    #[7] outlet temperature of the cooling water in ºC
]
pressure_8 = [
    pressure(1.01315, 'pa'), #[0] pressure of the first column
    pressure(1.01315, 'pa'), #[1] pressure of the second column
    pressure(2.5, 'pa')      #[2]  
]
##### CHARACTERIZATION OF VINASSE #####
d_vinasse = 1000 # density of vinasse in kg/m3
vlist_organic_x = np.array(list(map(lambda i: oxi_cod(
        vinasse['organic'][i]['c'],
        vinasse['organic'][i]['h'],
        vinasse['organic'][i]['o'],
        vinasse['organic'][i]['s'],
        vinasse['organic'][i]['n'],
        vinasse['organic'][i]['x_cod']*CODv,
        vinasse['organic'][i]['molar_mass']), vinasse['organic'].keys())))
vlist_MM_organic = np.array(list(map(lambda i: vinasse['organic'][i]['molar_mass'], vinasse['organic'].keys())))
vlist_organic_y = np.divide(vlist_organic_x, vlist_MM_organic)

vlist_inorganic_x = np.array(list(map(lambda i: vinasse['inorganic'][i]['x'], vinasse['inorganic'].keys())))
vlist_inorganic_x = np.append(vlist_inorganic_x, d_vinasse - (sum(vlist_organic_x) + sum(vlist_inorganic_x)))
vlist_MM_inorganic = np.array(list(map(lambda i: vinasse['inorganic'][i]['molar_mass'], vinasse['inorganic'].keys())))
vlist_MM_inorganic = np.append(vlist_MM_inorganic, prop_cp['water']['molar_mass'])
vlist_inorganic_y = np.divide(vlist_inorganic_x, vlist_MM_inorganic)
inorganic_dictv = list(vinasse['inorganic'].keys())
inorganic_dictv.append('water')

n_8_totalv = sum(vlist_organic_y) + sum(vlist_inorganic_y)
mm_vinasse = d_vinasse / n_8_totalv
vlist_organic_y = vlist_organic_y / n_8_totalv
vlist_inorganic_y = vlist_inorganic_y / n_8_totalv

vinasse_comp = dict.fromkeys(vinasse.keys(), None)
vinasse_comp['organic'] = dict(zip(vinasse['organic'].keys(), vlist_organic_y))
vinasse_comp['inorganic'] = dict(zip(inorganic_dictv, vlist_inorganic_y))

def chem_exergy_mix(d_in, index):
    try:
        if d_in[index] > 0:
            bch = d_in[index] * (vinasse['inorganic'][index]['chem_ex'] + R * t0 * np.log(d_in[index]))
        else:
            bch = 0
    except:
        if d_in[index] > 0:
            bch = d_in[index] * (prop_cp['water']['chem_x'] * prop_cp['water']['molar_mass'] + R * t0 * np.log(d_in[index]))
        else:
            bch = 0
    return bch

def ex_chem(d, index):
    if d[index] > 0:
        return d[index] * (prop_cp[index]['chem_x'] * prop_cp[index]['molar_mass'] + R * t0 * np.log(d[index]))
    else:
        return 0

x_8_wine = mass_to_molar_frac( #liquid molar fraction of the wine
    parameter_7[0], 
    prop_cp['ethanol']['molar_mass'], 
    prop_cp['water']['molar_mass']
    )
x_8_eth2 = mass_to_molar_frac( #liquid molar fraction of second grade ethanol
    parameter_8[0], 
    prop_cp['ethanol']['molar_mass'], 
    prop_cp['water']['molar_mass']
    )
x_8_eth = mass_to_molar_frac( #liquid molar fraction of ethanol
    parameter_8[1], 
    prop_cp['ethanol']['molar_mass'], 
    prop_cp['water']['molar_mass']
    )
x_8_vinasse = mass_to_molar_frac( #liquid molar fraction of vinasse
    parameter_8[2], 
    prop_cp['ethanol']['molar_mass'], 
    prop_cp['water']['molar_mass']
    )
x_8_flegmasse = mass_to_molar_frac( #liquid molar fraction of flegmasse
    parameter_8[3], 
    prop_cp['ethanol']['molar_mass'], 
    prop_cp['water']['molar_mass']
    )
x_8_flegma = mass_to_molar_frac( #liquid molar fraction of flegma
    parameter_8[4], 
    prop_cp['ethanol']['molar_mass'], 
    prop_cp['water']['molar_mass']
    )

n_e = m_7_out * parameter_7[0] / prop_cp['ethanol']['molar_mass'] #molar flow of ethanol in wine
n_wt = m_7_out * (1 - parameter_7[0]) / prop_cp['water']['molar_mass'] #molar flow of water in wine
n_v = ((n_wt * x_8_vinasse - n_e * (1 - x_8_vinasse)) * (parameter_8[5] + 1)) / (x_8_vinasse * (parameter_8[5] + 1) - x_8_flegma * parameter_8[5] - x_8_eth2)
n_eth2 = n_v * (1 - parameter_8[5] / 100)
m_8_eth2 = n_eth2 * x_8_eth2 * prop_cp['ethanol']['molar_mass'] / parameter_8[0]
n_flegma = n_v - n_eth2
m_8_flegma = n_flegma * x_8_flegma * prop_cp['ethanol']['molar_mass'] / parameter_8[4]
n_vinasse = n_e + n_wt - n_v
m_8_vinasse =  n_vinasse * x_8_vinasse * prop_cp['ethanol']['molar_mass'] / parameter_8[2]

n_eth = n_flegma * (1 - x_8_flegma / x_8_flegmasse) / (1 - x_8_eth / x_8_flegmasse)
m_8_eth = n_eth * x_8_eth * prop_cp['ethanol']['molar_mass'] / parameter_8[1]
n_flegmasse = n_flegma - n_eth
m_8_flegmasse =  n_flegmasse * x_8_flegmasse * prop_cp['ethanol']['molar_mass'] / parameter_8[3]

def activity_coef_ethanol_water(ye, s):
    a12 = 0.4104
    a21 = 0.7292
    ya = 1 - ye
    if s.lower() == 'e':
        act = np.exp(a12 * ((a21 * ya) / (a12 * ye + a21 * ya)) ** 2)
    else:
        act = np.exp(a21 * ((a21 * ye) / (a12 * ye + a21 * ya)) ** 2)
    return act
def str_ethanol_water(x_eth):
    return 'Ethanol[' + str(x_eth) + ']&Water[' + str(1 - x_eth) +']'

m_8_v2 = m_8_eth2 / (1 - parameter_8[5] / 100)
h_wine = cp('H', 'T', temperature(temperature_8[5], 'k'), 'P', pressure_8[0], 'water') / 1000 + h_mix_eth(x_8_wine, temperature(temperature_8[5], 'k'))
T_vg2 = cp('T', 'Q', 1, 'P', pressure_8[0], str_ethanol_water(x_8_eth2))
h_vg2 = cp('H', 'Q', 1, 'P', pressure_8[0], str_ethanol_water(x_8_eth2)) / 1000 + h_mix_eth(x_8_eth2, T_vg2)
h_vl2 = cp('H', 'T', temperature(temperature_8[4], 'k'),'P', pressure_8[0], str_ethanol_water(x_8_eth2)) / 1000 + h_mix_eth(x_8_eth2, temperature(temperature_8[4], 'k'))
T_flegma = cp('T', 'Q', 0, 'P', pressure_8[0], str_ethanol_water(x_8_flegma))
h_flegma = cp('H', 'Q', 0, 'P', pressure_8[0], str_ethanol_water(x_8_flegma)) / 1000 + h_mix_eth(x_8_flegma, T_flegma)
h_vinasse_r = cp('H', 'Q', 1, 'P', pressure_8[0], str_ethanol_water(x_8_vinasse)) / 1000 + h_mix_eth(x_8_vinasse, temperature(temperature_8[2], 'k'))
h_vinasse = cp('H', 'Q', 0, 'P', pressure_8[0], str_ethanol_water(x_8_vinasse)) / 1000 + h_mix_eth(x_8_vinasse, temperature(temperature_8[2], 'k'))
m_8_vinasse_r = (
    m_8_v2 * (h_vg2 - parameter_8[5] / 100 * h_vl2) + m_8_flegma * h_flegma + m_8_vinasse * h_vinasse - m_7_out * h_wine
    ) / (h_vinasse_r - h_vinasse)

T_vg = cp('T', 'Q', 1, 'P', pressure_8[0], str_ethanol_water(x_8_eth))
h_vg = cp('H', 'Q', 1, 'P', pressure_8[0], str_ethanol_water(x_8_eth)) / 1000 + h_mix_eth(x_8_eth, T_vg)
h_vl = cp('H', 'T', temperature(temperature_8[4], 'k'),'P', pressure_8[0], str_ethanol_water(x_8_eth)) / 1000 + h_mix_eth(x_8_eth, temperature(temperature_8[4], 'k'))
h_flegmasse_r = cp('H', 'Q', 1, 'P', pressure_8[0], str_ethanol_water(x_8_flegmasse)) / 1000 + h_mix_eth(x_8_flegmasse, temperature(temperature_8[3], 'k'))
h_flegmasse = cp('H', 'Q', 0, 'P', pressure_8[0], str_ethanol_water(x_8_flegmasse)) / 1000 + h_mix_eth(x_8_flegmasse, temperature(temperature_8[3], 'k'))
m_8_flegmasse_r = (
    m_8_eth * (parameter_8[6] * h_vg - (parameter_8[6] + 1) * h_vl) + m_8_flegmasse * h_flegmasse - m_8_flegma * h_flegma
    ) / (h_flegmasse_r - h_flegmasse)

m_8_vinasse_total = m_8_vinasse + m_8_flegmasse

h_wine60 = cp('H', 'T', temperature(temperature_8[0], 'k'),'P', pressure_8[0], 'water') / 1000 + h_mix_eth(x_8_wine, temperature(temperature_8[0], 'k'))
h_wine30 = cp('H', 'T', temperature(temperature_7[0], 'k'),'P', pressure_8[0], 'water') / 1000 + h_mix_eth(x_8_wine, temperature(temperature_7[0], 'k'))
h_int = (m_8_eth * h_vg - m_7_out * (h_wine60 - h_wine30)) / m_8_eth
h_w30 = cp('H', 'T', temperature(temperature_8[6], 'k'),'P', pressure_8[0], fluid) / 1000
h_w50 = cp('H', 'T', temperature(temperature_8[7], 'k'),'P', pressure_8[0], fluid) / 1000
h_wvapin = cp('H', 'Q', 1,'P', pressure_8[2], fluid) / 1000
h_wvapout = cp('H', 'Q', 0,'P', pressure_8[2], fluid) / 1000
m_w_reb1 = m_8_vinasse_r * (h_vinasse_r - h_vinasse) / (h_wvapin - h_wvapout)
m_w_reb2 = m_8_flegmasse_r * (h_flegmasse_r - h_flegmasse) / (h_wvapin - h_wvapout)

m_w_eth2 = m_8_v2 * (h_vg2 - h_vl2) / ((h_w50 - h_w30))
m_w_eth = (parameter_8[6] + 1) * m_8_eth * (h_vl - h_int) / ((h_w50 - h_w30))
h_vf_final = (m_8_vinasse_total * h_vinasse - m_7_out * (h_wine - h_wine60)) / m_8_vinasse_total
t_vf_final = cp('T', 'H', h_vf_final * 1000,'P', pressure_8[0], 'water')
def s_mix_eth2(h, ye, ae, yw, aw, T):
    #ye = molar fraction of ethanol
    #ae = activity coeficient of ethanol
    #yw = molar fraction of water
    #aw = activity coeficient of water
    #T = temperature in K
    return (h - mix_exergy(ae, aw, prop_cp['ethanol']['molar_mass'] * ye + prop_cp['water']['molar_mass'] * yw, ye, yw, T)) / T
m_8_coolwater = m_w_eth + m_w_eth2
m_8_reboiler = m_w_reb1 + m_w_reb2
m_ex_8 = [
    m_7_out * (ph_exergy( #[0]
        h_wine30,
        cp('S', 'T', temperature(temperature_7[0], 'k'), 'P', pressure_8[0], 'water') / 1000 + s_mix_eth2(
            h_wine30, 
            x_8_wine, 
            activity_coef_ethanol_water(x_8_wine, 'e'),
            1 - x_8_wine, 
            activity_coef_ethanol_water(x_8_wine, 'w'),
            temperature(temperature_7[0], 'k')
            ),
        cp('H', 'T', t0, 'P', p0, 'water') / 1000 + h_mix_eth(x_8_wine, t0),
        cp('S', 'T', t0, 'P', p0, 'water') / 1000 + s_mix_eth2(
            cp('H', 'T', t0, 'P', p0, 'water') / 1000 + h_mix_eth(x_8_wine, t0), 
            x_8_wine, 
            activity_coef_ethanol_water(x_8_wine, 'e'),
            1 - x_8_wine, 
            activity_coef_ethanol_water(x_8_wine, 'w'),
            t0
            )
        ) + (1 - parameter_7[0]) * prop_cp['water']['chem_x'] + parameter_7[0] * prop_cp['ethanol']['chem_x']
        ),
    m_8_coolwater * (ph_exergy( #[1]
        cp('H', 'T', temperature(temperature_8[6], 'k'),'P', pressure_8[0], fluid) / 1000,
        cp('S', 'T', temperature(temperature_8[6], 'k'),'P', pressure_8[0], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    m_8_reboiler * (ph_exergy( #[2]
        cp('H', 'Q', 1,'P', pressure_8[2], fluid) / 1000,
        cp('S', 'Q', 1,'P', pressure_8[2], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - m_8_coolwater * (ph_exergy( #[3]
        cp('H', 'T', temperature(temperature_8[7], 'k'),'P', pressure_8[0], fluid) / 1000,
        cp('S', 'T', temperature(temperature_8[7], 'k'),'P', pressure_8[0], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - m_8_reboiler * (ph_exergy( #[4]
        cp('H', 'Q', 0,'P', pressure_8[2], fluid) / 1000,
        cp('S', 'Q', 0,'P', pressure_8[2], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + prop_cp['water']['chem_x']),
    - m_8_eth2 * (ph_exergy( #[5]
        h_vl2,
        cp('S', 'T', temperature(temperature_8[4], 'k'),'P', pressure_8[0], str_ethanol_water(x_8_eth2)) / 1000 + s_mix_eth2(
            h_vl2, 
            x_8_eth2, 
            activity_coef_ethanol_water(x_8_eth2, 'e'),
            1 - x_8_eth2, 
            activity_coef_ethanol_water(x_8_eth2, 'w'),
            temperature(temperature_8[4], 'k')
            ),
        cp('H', 'T', t0,'P', p0, str_ethanol_water(x_8_eth2)) / 1000 + h_mix_eth(x_8_eth2, t0),
        cp('S', 'T', t0,'P', p0, str_ethanol_water(x_8_eth2)) / 1000 + s_mix_eth2(
            cp('H', 'T', t0,'P', p0, str_ethanol_water(x_8_eth2)) / 1000 + h_mix_eth(x_8_eth2, t0), 
            x_8_eth2, 
            activity_coef_ethanol_water(x_8_eth2, 'e'),
            1 - x_8_eth2, 
            activity_coef_ethanol_water(x_8_eth2, 'w'),
            t0
            )
        ) + (1 - parameter_8[0]) * prop_cp['water']['chem_x'] + parameter_8[0] * prop_cp['ethanol']['chem_x']
        ),
    - m_8_eth * (ph_exergy( #[6]
        h_vl,
        cp('S', 'T', temperature(temperature_8[4], 'k'),'P', pressure_8[0], str_ethanol_water(x_8_eth)) / 1000 + s_mix_eth2(
            h_vl, 
            x_8_eth, 
            activity_coef_ethanol_water(x_8_eth, 'e'),
            1 - x_8_eth, 
            activity_coef_ethanol_water(x_8_eth, 'w'),
            temperature(temperature_8[4], 'k')
            ),
        cp('H', 'T', t0,'P', p0, str_ethanol_water(x_8_eth)) / 1000 + h_mix_eth(x_8_eth, t0),
        cp('S', 'T', t0,'P', p0, str_ethanol_water(x_8_eth)) / 1000 + s_mix_eth2(
            cp('H', 'T', t0,'P', p0, str_ethanol_water(x_8_eth)) / 1000 + h_mix_eth(x_8_eth, t0), 
            x_8_eth, 
            activity_coef_ethanol_water(x_8_eth, 'e'),
            1 - x_8_eth, 
            activity_coef_ethanol_water(x_8_eth, 'w'),
            t0
            )
        ) + (1 - parameter_8[1]) * prop_cp['water']['chem_x'] + parameter_8[1] * prop_cp['ethanol']['chem_x']
        ),
    - m_8_vinasse_total * (ph_exergy( #[7]
        h_vf_final,
        cp('S', 'T', t_vf_final,'P', pressure_8[0], fluid) / 1000,
        cp('H', 'T', t0,'P', p0, fluid) / 1000,
        cp('S', 'T', t0,'P', p0, fluid) / 1000
        ) + parameter_8[7] * CODv * mm_vinasse + sum(list(map(lambda i: chem_exergy_mix(vinasse_comp['inorganic'], i), vinasse_comp['inorganic'].keys()))) / mm_vinasse
    )
]

print('Subsystem 8 - ETHANOL DISTILLATION - exergy destruction: {:.2f} MW'.format(exergy_destruction(0, 0, m_ex_8) / 1000))
#####################################################################################################
###################################### BIOGAS PRODUCTION (9) ########################################
#####################################################################################################
parameter_9 = [
    0.3,   #[0] kg of bicarbonate/kg of COD. Source: I. Alves, “Caracterização de grânulos 
#de reator UASB empregado no processamento de vinhaça,” text, Universidade de São Paulo, 2015. doi: 10.11606/D.18.2015.tde-
### ---> Source: C. A. de Lemos Chernicharo, Anaerobic reactors. London: IWA Publ. [u.a.], 2007. ###    
    0.75,  #[1] removal efficiency of organic matter
    0.03,  #[2] yield coefficient for methanogenic microorganisms
    0.15,  #[3] yield coefficient for acidogenic microorganisms
    0.05,  #[4] solids concentration in the sludge (fraction)
    0.2,   #[5] yield or solids production coefficient (kgTSS/kgCOD)
### <--- Source: C. A. de Lemos Chernicharo, Anaerobic reactors. London: IWA Publ. [u.a.], 2007. ###
    10,    #[6] kg of NaOH /m3 of solution
    3.5,   #[7] kg of NaOH/kgH2S
    0.0002 #[8] final concentration of H2S in biogas
]
pressure_9 = [
    p0 #[0] pressure in the anaerobic reactor (Pa)
]
temperature_9 = [
    35, #[0] temperature in the anaerobic reactor in C
    15  #[1] temperature of naoh solution
]

##### CHARACTERIZATION OF BICARBONATE STREAM #####
m_9_bicarbonate = parameter_9[1] *CODv * m_8_vinasse_total / d_vinasse #kg of bicarbonate/s. Source: I. Alves, “Caracterização de grânulos 
#de reator UASB empregado no processamento de vinhaça,” text, Universidade de São Paulo, 2015. doi: 10.11606/D.18.2015.tde-
##### CHARACTERIZATION OF BIOGAS #####
v_methane = v_ch4(pressure_9[0], temperature(temperature_9[0], 'k'), parameter_9[1], parameter_9[2], parameter_9[3], vinasse['inorganic']['sulfate']['x'], CODv) #m3 of methane/m3 of vinasse
v_biogas = v_methane / biogas['methane'] #m3 of biogas/m3 of vinasse
v_co2 = v_biogas * biogas['carbon_dioxide'] #m3 of CO2/m3 of vinasse
v_h2s = v_biogas * biogas['hydrogen_sulfide'] #m3 of H2S/m3 of vinasse
heos_biogas = 'HEOS::CH4['+str(biogas['methane'])+']&CO2['+str(biogas['carbon_dioxide'])+']&H2S['+str(biogas['hydrogen_sulfide'])+']' #string that describes the composition of biogas mixture
d_biogas = cp('D','T', t0, 'P', p0, heos_biogas)
m_9_biogas = v_biogas * d_biogas * m_8_vinasse_total / d_vinasse #kg/s of biogas 

##### CHARACTERIZATION OF SLUDGE #####
sludge = parameter_9[5] * CODv / parameter_9[4] #kg of sludge/m^3 of vinasse
m_9_sludge = sludge * m_8_vinasse_total / d_vinasse #kg of sludge/s

##### CHARACTERIZATION OF EFFLUENT #####
m_9_effluent = m_8_vinasse_total - m_9_biogas - m_9_sludge #mass balance to calculate the quantity of effluent in kg/s
d_eff = {'organic':{}, 'inorganic':{}} #dictionary that saves the concentration of organic and inorganic matter in kg/m^3 of vinasse
elist_organic_x = np.array(list(map(lambda i: oxi_cod(
    vinasse['organic'][i]['c'],
    vinasse['organic'][i]['h'],
    vinasse['organic'][i]['o'],
    vinasse['organic'][i]['s'],
    vinasse['organic'][i]['n'],
    vinasse['organic'][i]['x_cod']*CODv * (1 - parameter_9[1] - parameter_9[4]),
    vinasse['organic'][i]['molar_mass']), vinasse['organic'].keys())))
elist_organic_y = np.divide(elist_organic_x, vlist_MM_organic)

def bac_food(eff, bac, index):
    try:
        res = eff['inorganic'][index]['x'] - nutritional_req(CODv, 1.88, 2.21, bac['macronutrients'][index] / 1000)
        if res > 0:
            return res
        else:
            return 0
    except:
        return eff['inorganic'][index]['x']

elist_inorganic_x = np.array(list(map(lambda i: bac_food(vinasse, bacteria, i),vinasse['inorganic'].keys())))
elist_inorganic_x = np.append(elist_inorganic_x, vlist_inorganic_x[-1] - (1 - parameter_9[4]) * sludge)
elist_inorganic_y = np.divide(elist_inorganic_x, vlist_MM_inorganic)

n_9_totale = sum(elist_organic_y) + sum(elist_inorganic_y)
elist_organic_y = elist_organic_y / n_9_totale
elist_inorganic_y = elist_inorganic_y / n_9_totale

d_eff = dict.fromkeys(vinasse.keys(), None)
d_eff['organic'] = dict(zip(vinasse['organic'].keys(), elist_organic_y))
d_eff['inorganic'] = dict(zip(inorganic_dictv, elist_inorganic_y))

##### EVALUATION OF HEAT LOSSES IN VINASSE STORAGE TANK #####
t_9_q = list(range(int(temperature(20, 'k')), int(temperature(100, 'k'))))
cp_9_q = list(map(lambda x: cp('C', 'T', x, 'P', pressure_9[0], fluid) / 1000, t_9_q))
cof_9 = list(np.polyfit(t_9_q, cp_9_q, 1))
q_ex_9 = int_first(cof_9, len(cof_9), 0, 0, t_vf_final, temperature(temperature_9[0], 'k')) + int_second(cof_9, len(cof_9) - 1, 0, 0, t_vf_final, temperature(temperature_9[0], 'k'))

### BIOGAS CLEANING ###
m_ch4 = v_methane  * cp('D','T', temperature(temperature_9[0], 'k'), 'P', pressure_9[0], 'CH4') * m_8_vinasse_total / d_vinasse #kg of methane/s
m_co2 = v_co2 * cp('D','T', temperature(temperature_9[0], 'k'), 'P', pressure_9[0], 'CO2') * m_8_vinasse_total / d_vinasse
m_h2s_in = v_h2s * cp('D', 'T', temperature(temperature_9[0], 'k'), 'P', pressure_9[0], 'H2S') * m_8_vinasse_total / d_vinasse
m_h2s_out = parameter_9[8] * (m_ch4 + m_co2) / (1 - parameter_9[8]) #mass outflow of h2s in biogas
v_h2s_out = m_h2s_out / cp('D','T',temperature(temperature_9[0], 'k'), 'P', pressure_9[0], 'H2S')
#given that, the density of NaOH(aq) at 15ºC is about 1.01. 
naoh_in = parameter_9[7] * (m_h2s_in - m_h2s_out) #quantity of NaOH that must be used to neutralize H2S
d_solution = 1010 #kg/m^3

m_biogas_out = m_ch4 + m_co2 + m_h2s_out 
v_biogas_out = v_methane + v_co2 + v_h2s_out
mol_biogas_out = m_ch4 / prop_cp['methane']['molar_mass'] + m_co2 / prop_cp['carbon_dioxide']['molar_mass'] + m_h2s_out / prop_cp['hydrogen_sulfide']['molar_mass']
biogas_out = {
    'methane': v_methane / v_biogas_out,
    'carbon_dioxide': v_co2 / v_biogas_out,
    'hydrogen_sulfide': v_h2s_out / v_biogas_out
}
mm_biogas_out = m_biogas_out / mol_biogas_out
heos_biogas_out = 'HEOS::CH4[' + str(biogas_out['methane']) + ']&CO2[' + str(biogas_out['carbon_dioxide']) + ']&H2S[' + str(biogas_out['hydrogen_sulfide']) + ']' 
water_in = naoh_in / parameter_9[6] * d_solution * (1 - parameter_9[6] / d_solution) #water to dissolve the NaOH 
keys_solution = ['water', 'sodium_hydroxide', 'sodium_sulfide']
y_solution_in = np.array([water_in / prop_cp['water']['molar_mass'], naoh_in / prop_cp['water']['molar_mass'], 0])
mm_solution_in = sum(y_solution_in)
y_solution_in = np.divide(y_solution_in, mm_solution_in)
solution_in = dict(zip(keys_solution, y_solution_in))

water_out = water_in + naoh_in * prop_cp['water']['molar_mass'] / prop_cp['sodium_hydroxide']['molar_mass']
naoh_out = (parameter_9[7] - 2  * prop_cp['sodium_hydroxide']['molar_mass'] / prop_cp['hydrogen_sulfide']['molar_mass']) * (m_h2s_in - m_h2s_out)
na2s_out = prop_cp['sodium_sulfide']['molar_mass'] / prop_cp['hydrogen_sulfide']['molar_mass'] * (m_h2s_in - m_h2s_out)

y_solution_out = np.array([water_out / prop_cp['water']['molar_mass'], naoh_out / prop_cp['water']['molar_mass'], na2s_out / prop_cp['sodium_sulfide']['molar_mass']])
mm_solution_out = sum(y_solution_out)
y_solution_out = np.divide(y_solution_out, mm_solution_out)
solution_out = dict(zip(keys_solution, y_solution_out))

q_9 = [
    - m_8_vinasse_total * q_ex_9
]   
m_ex_9 = [
     - m_ex_8[7], #[0]
    m_9_bicarbonate * prop_cp['bicarbonate']['chem_x'], #[1]
    (water_in + naoh_in) * sum(list(map(lambda i: ex_chem(solution_in, i), solution_in.keys()))) / prop_cp['water']['molar_mass'], #[2]
    - m_biogas_out * (sum(list(map(lambda i: ex_chem(biogas_out, i), biogas_out.keys()))) / mm_biogas_out + ph_exergy( #[3]
        cp('H', 'T', temperature(temperature_9[0], 'k'), 'P', pressure_9[0], heos_biogas_out) / 1000,
        cp('S', 'T', temperature(temperature_9[0], 'k'), 'P', pressure_9[0], heos_biogas_out) / 1000,
        cp('H', 'T', t0, 'P', pressure_9[0], heos_biogas_out) / 1000, 
        cp('S', 'T', t0, 'P', pressure_9[0], heos_biogas_out) / 1000)
    ),
    - (water_out + naoh_out + na2s_out) * sum(list(map(lambda i: ex_chem(solution_out, i), solution_out.keys()))) / prop_cp['water']['molar_mass'], #[4]
    - m_9_effluent * (ph_exergy(
        cp('H', 'T', temperature(temperature_9[0], 'k'), 'P', p0, fluid) / 1000,
        cp('S', 'T', temperature(temperature_9[0], 'k'), 'P', p0, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000, 
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
        ) + parameter_8[7] * CODv*(1 - parameter_9[1] - parameter_9[4]) * prop_cp['water']['molar_mass'] + sum(list(map(lambda i: chem_exergy_mix(d_eff['inorganic'], i), d_eff['inorganic'].keys()))) / prop_cp['water']['molar_mass']),
    - m_9_sludge * (ph_exergy(
        cp('H', 'T', temperature(temperature_9[0], 'k'), 'P', p0, fluid) / 1000,
        cp('S', 'T', temperature(temperature_9[0], 'k'), 'P', p0, fluid) / 1000,
        cp('H', 'T', t0, 'P', p0, fluid) / 1000, 
        cp('S', 'T', t0, 'P', p0, fluid) / 1000
        ) + parameter_8[7] * CODv * parameter_9[4] * prop_cp['water']['molar_mass'] + (1 - parameter_9[4]) * prop_cp['water']['chem_x']
    )
]

print('Subsystem 9 - BIOGAS PRODUCTION - exergy destruction: {:.2f} MW'.format(exergy_destruction(q_9, 0, m_ex_9) / 1000))
#####################################################################################################
#################################### COGENERATION SYSTEM (10) #######################################
#####################################################################################################
parameter_10 = [

]
temperature_10 = [
    750, #[0] temperature of the combustion in C
    170, #[1] temperature of the flue gases in C
    70   #[2]
]
pressure_10 = [
    80 # [0] in bar
]
#function that calculates the number of mols of co2 and water formated, as well as the mols of oxygen and nitrogen needed in the 
#combustion of dry bagasse and methane
def mols_comb_boiler(n_bagdry, ycell, yhemi, ylig, n_biogas, ych4, yo2):
    #n_bagdry = dry bagasse number of mols
    #ycell = molar fraction of cellulose in dry bagasse
    #yhemi = molar fraction of hemicellulose in dry bagasse
    #ylig = molar fraction of lignin in dry bagasse
    #n_biogas = biogas number of mols
    #ych4 = molar fraction of methane in biogas
    #yo2 = molar fraction of oxygen in the mixture of O2 and N2
    n_co2_f = 6 * n_bagdry * ycell + 6 * n_bagdry * yhemi + 10 * n_bagdry * ylig + n_biogas * ych4 #number of mols of carbon dioxide
    #formed in the oxicombustion of bagasse and biogas
    n_wf = 5 * n_bagdry * ycell + 5 * n_bagdry * yhemi + 5.75 * n_bagdry * ylig + 2 * n_biogas * ych4 #number of mols of water
    #formed in the oxicombustion of bagasse and biogas
    n_o2 = (n_wf + 2 * n_co2_f - (5 * n_bagdry * ycell + 5 * n_bagdry * yhemi + 3.9 * n_bagdry * ylig)) / 2 #number of mols of oxygen
    #needed to oxicombust bagasse and biogas
    nn2 = n_o2 / yo2 * (1 - yo2)
    return [n_co2_f, n_wf, n_o2, nn2]
#function that calculates the dry bagasse enthalpy of formation
def enthalpy_f_drybagasse(ycell, yhemi, ylig):
    #ycell = molar fraction of cellulose in dry bagasse
    #yhemi = molar fraction of hemicellulose in dry bagasse
    #ylig = molar fraction of lignin in dry bagasse
    return ycell * prop_cp['cellulose']['enthalpy_f'] + yhemi * prop_cp['hemicellulose']['enthalpy_f'] + ylig * prop_cp['lignin']['enthalpy_f']
#function that calculates the mols of water needed to recirculate in the boiler
def n_rec(n_bagdry, ycell, yhemi, ylig, n_biogas, ych4, yco2, n_co2_f, n_wf, nn2, Tcomb):
    dlt_hco2 = (cp('Hmolar', 'T', Tcomb, 'P', p0, 'CO2') - cp('Hmolar', 'T', t0, 'P', p0, 'CO2')) / 1000
    dlt_hh2o = (cp('Hmolar', 'T', Tcomb, 'P', p0, fluid) - cp('Hmolar', 'T', t0, 'P', p0, fluid)) / 1000
    dlt_hn2 = (cp('Hmolar', 'T', Tcomb, 'P', p0, 'N2') - cp('Hmolar', 'T', t0, 'P', p0, 'N2')) / 1000
    
    return (n_co2_f * prop_cp['carbon_dioxide']['enthalpy_f'] + (n_co2_f + n_biogas * yco2) * dlt_hco2 + n_wf * (prop_cp['water']['enthalpy_f'] + dlt_hh2o) + nn2 * dlt_hn2 - n_bagdry * enthalpy_f_drybagasse(ycell, yhemi, ylig) - n_biogas * ych4 * prop_cp['methane']['enthalpy_f']) / ((cp('Hmolar', 'Q', 0, 'T', t0, fluid) - cp('Hmolar', 'Q', 1, 'T', t0, fluid)) / 1000 - dlt_hh2o)
###BOILER###
mm_dry_bagasse = 1 / (bagasse['fibres'][1]['cellulose'] / prop_cp['cellulose']['molar_mass'] + bagasse['fibres'][1]['hemicellulose'] / prop_cp['hemicellulose']['molar_mass'] + bagasse['fibres'][1]['lignin'] / prop_cp['lignin']['molar_mass'])
n_bagdry = m_bagasse * (1 - bagasse['water']) / mm_dry_bagasse
ycell = bagasse['fibres'][1]['cellulose'] * prop_cp['cellulose']['molar_mass'] / mm_dry_bagasse
yhemi = bagasse['fibres'][1]['hemicellulose'] * prop_cp['hemicellulose']['molar_mass'] / mm_dry_bagasse
ylig = bagasse['fibres'][1]['lignin'] * prop_cp['lignin']['molar_mass'] / mm_dry_bagasse

prod_n = mols_comb_boiler(n_bagdry, ycell, yhemi, ylig, mol_biogas_out, biogas_out['methane'], 1)
nrec = n_rec(
    n_bagdry, 
    ycell, 
    yhemi, 
    ylig, 
    mol_biogas_out, 
    biogas['methane'], 
    biogas['carbon_dioxide'],
    prod_n[0],
    prod_n[1],
    prod_n[3],
    temperature(temperature_10[0], 'k')
)
n_h2o = nrec + prod_n[1] 
n_co2 = prod_n[0] + mol_biogas_out * biogas_out['carbon_dioxide']
n_ftotal = n_h2o + n_co2 + prod_n[3]
yf_h2o = n_h2o / n_ftotal
yf_co2 = n_co2 / n_ftotal
yf_n2 = prod_n[3] / n_ftotal
m_rec = nrec * prop_cp['water']['molar_mass'] - m_bagasse * bagasse['water']
m_o2 = prod_n[2] * prop_cp['oxygen']['molar_mass']
m_co2 = n_co2 * prop_cp['carbon_dioxide']['molar_mass']
m_h2o = n_h2o * prop_cp['water']['molar_mass']
m_n2 = prod_n[3] * prop_cp['nitrogen']['molar_mass']

if yf_n2 > 0:
    heos_fluegases = 'HEOS::WATER[' + str(yf_h2o) + ']&CO2[' + str(yf_co2) + ']&N2[' + str(yf_n2)+']'
else:
    heos_fluegases = 'HEOS::WATER[' + str(yf_h2o) + ']&CO2[' + str(yf_co2) + ']'

m_fluegases = m_co2 + m_h2o + m_n2
print(m_fluegases * 3.6)
q_flue = (cp('H', 'T', temperature(temperature_10[0], 'k'), 'P', pressure(40, 'pa'), heos_fluegases) - cp('H', 'T', temperature(temperature_10[1], 'k'), 'P', pressure(40, 'pa'), heos_fluegases)) / 1000
Q_flue = m_fluegases * q_flue

#DEWATERING CYCLONE#
T_dw = cp('T', 'Q', 0, 'P', p0 * (yf_co2 + yf_n2), fluid) - 10 #temperature of the outlet water
m_fluegases_out = m_co2 + m_n2
yf_co2out = round(yf_co2 / (yf_co2 + yf_n2), 2)
yf_n2out = round(yf_n2 / (yf_co2 + yf_n2), 2)
p0_dwout = p0 * (yf_co2 + yf_n2)
if yf_n2out > 0:
    heos_fluegases_out = 'HEOS::CO2[' + str(yf_co2out) + ']&N2[' + str(yf_n2out) + ']'
    q_dwn2 = (cp('H', 'T', temperature(temperature_10[1], 'k'), 'P', p0 * yf_n2, 'N2') - cp('H', 'T', T_dw, 'P', p0_dwout * yf_n2, 'N2')) / 1000
    t_10_bn2 = list(range(int(T_dw - 10), int(temperature(temperature_10[1] + 10, 'k'))))
    cp_10_bn2 = list(map(lambda x: cp('C', 'T', x, 'P', p0_dwout * yf_n2out, 'N2') / 1000, t_10_bn2))
    cof_10_bn2 = list(np.polyfit(t_10_bn2, cp_10_bn2, 1))
    b_n2 = int_first(cof_10_bn2, len(cof_10_bn2), 0, 0, temperature(temperature_10[1], 'k'), T_dw) + int_second(cof_10_bn2, len(cof_10_bn2) - 1, 0, 0, temperature(temperature_10[1], 'k'), T_dw)
else:
    heos_fluegases_out = 'CO2'
    q_dwn2 = 0
    b_n2 = 0

q_dwh2o = (cp('H', 'T', temperature(temperature_10[1], 'k'), 'P', p0 * yf_h2o, fluid) - cp('H', 'T', T_dw, 'P', p0_dwout, fluid)) / 1000
q_dwco2 = (cp('H', 'T', temperature(temperature_10[1], 'k'), 'P', p0 * yf_co2, 'CO2') - cp('H', 'T', T_dw, 'P', p0_dwout * yf_co2out, 'CO2')) / 1000

Q_dw = m_h2o * q_dwh2o + m_co2 * q_dwco2 + m_n2 * q_dwn2
m_h2oout = m_h2o - m_rec

#specific exergy of dw heat
t_10_bh2o1 = list(range(int(T_dw + 10), int(temperature(temperature_10[1], 'k'))))
cp_10_bh2o1 = list(map(lambda x: cp('C', 'T', x, 'P', p0 * yf_h2o, fluid) / 1000, t_10_bh2o1))
cof_10_bh2o1 = list(np.polyfit(t_10_bh2o1, cp_10_bh2o1, 2))
t_10_bh2o2 = list(range(int(T_dw - 10), int(T_dw + 10)))
cp_10_bh2o2 = list(map(lambda x: cp('C', 'T', x, 'P', p0_dwout, fluid) / 1000, t_10_bh2o2))
cof_10_bh2o2 = list(np.polyfit(t_10_bh2o2, cp_10_bh2o2, 2))
b_h2o = int_first(cof_10_bh2o1, len(cof_10_bh2o1), 0, 0, temperature(temperature_10[1], 'k'), T_dw + 10) + int_second(cof_10_bh2o1, len(cof_10_bh2o1)-1, 0, 0, temperature(temperature_10[1], 'k'), T_dw + 10) + (1 - t0 / (T_dw + 10)) * (cp('H', 'Q', 0, 'T', T_dw + 10, fluid) - cp('H', 'Q', 1, 'T', T_dw + 10, fluid)) / 1000 + int_first(cof_10_bh2o2, len(cof_10_bh2o2), 0, 0, T_dw + 10, T_dw) + int_second(cof_10_bh2o2, len(cof_10_bh2o2)-1, 0, 0, T_dw + 10, T_dw) 

t_10_bco2 = list(range(int(T_dw - 10), int(temperature(temperature_10[1] + 10, 'k'))))
cp_10_bco2 = list(map(lambda x: cp('C', 'T', x, 'P', p0_dwout * yf_co2out, 'CO2') / 1000, t_10_bco2))
cof_10_bco2 = list(np.polyfit(t_10_bco2, cp_10_bco2, 1))
b_co2 = int_first(cof_10_bco2, len(cof_10_bco2), 0, 0, temperature(temperature_10[1], 'k'), T_dw) + int_second(cof_10_bco2, len(cof_10_bco2)-1, 0, 0, temperature(temperature_10[1], 'k'), T_dw)

class ThermoPoint:
    def __init__(self, p1, s1, p2, s2, stm):
        self.fld = stm
        self.t = cp('T', s1, p1, s2, p2, stm)
        self.p = cp('P', s1, p1, s2, p2, stm)
        self.q = cp('Q', s1, p1, s2, p2, stm)
        self.h = cp('H', s1, p1, s2, p2, stm)
        self.s = cp('S', s1, p1, s2, p2, stm)

class WorkCCS(ThermoPoint):
    def work(self, y):
        P1 = WorkCCS(T_dw, 'T', p0_dwout * y, 'P', self.fld)
        P3 = WorkCCS(temperature(temperature_10[2], 'k'), 'T', self.p, 'P', self.fld)
        P4 = WorkCCS(P3.s, 'S', pressure(pressure_10[0], 'pa'), 'P', self.fld)
        w12 = self.h - P1.h
        w34 = P4.h - P3.h
        return w12 + w34

list_pressure = list(range(int(p0_dwout + 10000), pressure(11, 'pa'), 10000))
co2_flue = WorkCCS(T_dw, 'T', p0_dwout * yf_co2out, 'P', 'CO2')
co2_out = WorkCCS(temperature(temperature_10[2], 'k'), 'T', pressure(pressure_10[0], 'pa'), 'P', 'CO2')
list_wk1 = list(map(lambda pt: WorkCCS(pt * yf_co2out, 'P', co2_flue.s, 'S', co2_flue.fld), list_pressure))
if yf_n2out > 0:
    n2_flue = WorkCCS(T_dw, 'T', p0_dwout * yf_n2out, 'P', 'N2')
    n2_out = WorkCCS(temperature(temperature_10[2], 'k'), 'T', pressure(pressure_10[0], 'pa'), 'P', 'N2')
    list_wk2 = list(map(lambda pt: WorkCCS(pt * yf_n2out, 'P', n2_flue.s, 'S', n2_flue.fld), list_pressure))
    wk_total = np.array(list(map(lambda comp1, comp2: (m_co2 * comp1.work(yf_co2out) + m_n2 * comp2.work(yf_n2out)) / 1000, list_wk1, list_wk2)))
    indice_min = np.where(wk_total == np.amin(wk_total))[0][0]
else:
    wk_total = np.array(list(map(lambda comp1: m_co2 * comp1.work(yf_co2out) / 1000, list_wk1)))
    indice_min = np.where(wk_total == np.amin(wk_total))[0][0]

plt.plot(np.divide(np.array(list_pressure), 1e6), np.divide(wk_total, 10000))
plt.title('Power demanded in function of Pint')
plt.xlabel('Pressure [MPa]')
plt.ylabel('Power in flue gases compression [MW]')
plt.show()

###SEPARATION PROCESS###
m_10_air = prod_n[2] / air['oxygen'] * prop_cp['air']['molar_mass']
W_sep = R * t0 * sum(list(map(lambda c: air[c] * np.log(air[c]), air))) / prop_cp['air']['molar_mass']

###COGENERATION###
parameter_11 = [
    50000 #[0] total power exported in kW
]
temperature_11 = [
    30,  #[0] pump inlet temperature of the Rankine cycle
    25,  #[1] compressor inlet temperature of the T-CO2 Brayton cycle
    500, #[2] turbine inlet temperature of the Rankine cycle
    700  #[3] turbine inlet temperature of the T-CO2 Brayton cycle
]
pressure_11 = [
    40,
    2.5
]
m_10_rankine = massflow_4[0] + m_6_steam + m_8_reboiler #low pressure steam that leaves the rankine cycle
#thermodynamic points of the Rankine cyle
#p_11_1 = ThermoPoint(pressure(pressure_11[0], 'pa'), 'P', temperature(temperature_11[3], 'k'), 'T', fluid) #turbine inlet
#p_11_2 = ThermoPoint(pressure(pressure_11[1], 'pa'), 'P', p_11_1.s, 'S', fluid) #turbine outlet
#p_11_3 = ThermoPoint(p0, 'P', temperature(temperature_11[0], 'k'), 'T', fluid) #water that enters the cycle
#p_11_6 = ThermoPoint(p0, 'P', 0, 'Q', fluid) #pump inlet
#p_11_7 = ThermoPoint(pressure(pressure_11[1], 'pa'), 'P', p_11_6.s, 'S', fluid) #turbine outlet
#m_11_rnkcycle = Q_flue / ((p_11_1.h - p_11_7.h) / 1000)
p_11_1 = ThermoPoint(p0, 'P', 0, 'Q', fluid)
p_11_2 = ThermoPoint(pressure(pressure_11[0], 'pa'), 'P', p_11_1.s, 'S', fluid)
print(p_11_2.t - 273.15)
h3 = p_11_2.h + Q_flue * 1000 / m_10_rankine
p_11_3 = ThermoPoint(pressure(pressure_11[0], 'pa'), 'P', h3, 'H', fluid)
print(p_11_3.t - 273.15)
p_11_4 = ThermoPoint(pressure(pressure_11[1], 'pa'), 'P', p_11_3.s, 'S', fluid)

Wtot = (p_11_3.h - p_11_4.h) / 1000 * m_10_rankine
print(cp('T', 'P', pressure(22, 'pa'), 'Q', 1, fluid) - 273.15)

###T-CO2 -BRAYTON###
temperature_brayton = [
    750
]
pressure_brayton = [
    150,
    2.5
]
p_b_1 = ThermoPoint(temperature(temperature_brayton[0], 'k'), 'T', pressure(pressure_brayton[0], 'pa'), 'P', 'CO2')
p_b_2 = ThermoPoint(p_b_1.s, 'S', pressure(pressure_brayton[1], 'pa'), 'P', 'CO2')
print(p_b_2.t - 273.15, p_b_1.q)
