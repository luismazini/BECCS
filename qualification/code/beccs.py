import numpy as np
import random
from CoolProp.CoolProp import PropsSI as cp
import json
from scipy.optimize import newton

###FUNCTIONS###
def ph_exergy(h: float, s: float, h0: float, s0:float) -> float:
    '''
    Calculates the physical exergy in kJ/kg

    :param float h: enthalpy in kJ/kg
    :param float s: entropy in kJ/kgK
    :param float h0: dead-state enthalpy in kJ/kg
    :param float s0: dead-state entropy in kJ/kgK
    :return: the physical exergy in kJ/kg.
    '''
    return h - h0 - t0*(s-s0)

def exergy_destruction(q_ex: list, w_ex: list, m_ex: list) -> float:
    '''
    Calculates the exergy destruction in kW

    :param list q_ex: list of exergies from heat
    :param list w_ex: list of exergies from power
    :param list m_ex: list of exergies from massflow
    :return: the exergy destruction in kW
    '''
    q_sum, w_sum, m_sum = 0, 0, 0 #initiate the dummy variables to sum the exergy contents
    if q_ex != 0:
        #loop that calculates the sum of heat exergy contents
        q_sum = sum(q_ex)
    if w_ex != 0:
        #loop that calculates the sum of work exergy contents   
            w_sum = sum(w_ex)
    if m_ex != 0:
        #loop that calculates the sum of mass flow exergy contents   
            m_sum = sum(m_ex)
    return q_sum - w_sum + m_sum

def mass_flow(m: float, s: str) -> float:
    '''
    Convert ton/h into kg/s and vice-versa

    :param float m: mass flow in ton/h or kg/s
    :param string s: if you pass "kg", the function will convert it to kg/s. However, if you pass "ton", it will convert it to ton/h.
    :return: converted mass flow
    '''
    if s.lower() == 'kg':
        m = m * 1000 / 3600
    elif s.lower() == 'ton':
        m = m * 3.6
    return m

def temperature(t: float, s: str) -> float:
    '''
    Convert ºC into K and vice-versa

    :param float t: temperature in ºC or K
    :param string s: if you pass "c", the function will convert it to Celsius. However, if you pass "k", it will convert it to K.
    :return: converted temperature
    '''
    if s.lower() == 'c':
        t += -273.15
    elif s.lower() == 'k':
        t += 273.15
    return t

def pressure(p: float, s: str) -> float:
    '''
    Convert Bar into Pa and vice-versa

    :param float t: temperature in ºC or K
    :param string s: if you pass "c", the function will convert it to Celsius. However, if you pass "k", it will convert it to K.
    :return: converted mass flow
    '''
    if s.lower() == 'bar':
        p = p / 1e5
    elif s.lower() == 'pa':
        p = p * 1e5
    return p

def integral(f: list, s: float, i: int, t_in: float, t_out: float) -> float:
    '''
    Calculate the integral of a polynomial function

    :param list f: coeficients of the Cp function
    :param float s: the result of the integration
    :param int i: the iterator variable
    :param float t_in: The inlet temperature
    :param float t_out: The outlet temperature
    :return: the integration
    '''
    c = len(f) - 1 - i 
    if i <= len(f) - 1:
        s += f[i] / (c+1) * (t_out**(c+1) - t_in**(c+1))
        i += 1
        return integral(f, s, i, t_in, t_out)
    else:
        return s

def cp_function(f: list, s: float, i: int, t_in: float, t_out: float) -> float:
    '''
    Calculate the cp polynomial function

    :param list f: coeficients of the Cp function
    :param float s: the result of the function
    :param int i: the iterator variable
    :param float t_in: the inlet temperature
    :param float t_out: the outlet temperature
    :return: the polynomial result
    '''
    c = len(f) - 1 - i
    if i <= len(f)-1:
        if c > 0:
            s+=f[i] * (t_out ** c - t_in ** c)
            i+=1
            return cp_function(f, s, i, t_in, t_out)
        else:
            s+=f[i]
            return s
    else:
        return s

def int_first(f: list, s: float, i: int, t_in: float, t_out: float) -> float:
    '''
    Calculate the first part of the integral that will calculate the exergy content of heat in kW.

    :param list f: coeficients of the polynomium that represents the cp function.
    :param float s: the result of the function
    :param int i: the iterator variable
    :param float t_in: the inlet temperature
    :param float t_out: the outlet temperature
    :return: the polynomial result
    '''
    c = len(f) - 1 - i 
    if i <= len(f) - 1:
        if c > 0:
            s += f[i] / c * (t_out**c - t_in**c)
            i += 1
            return int_first(f, s, i, t_in, t_out)
        else:
            return s
    else:
        return s

def int_second(f: list, s: float, i: int, t_in: float, t_out: float) -> float:
    '''
    Calculate the second part of the integral that will calculate the exergy content of heat in kW.

    :param list f: coeficients of the polynomium that represents the cp function.
    :param float s: the result of the function
    :param int i: the iterator variable
    :param float t_in: the inlet temperature
    :param float t_out: the outlet temperature
    :return: the polynomial result
    '''
    c = len(f) - 2 - i
    if i <= len(f) - 1:
        if c > 0:
            s += -t0 * f[i] / c * (t_out**c - t_in**c)
            i += 1
            return int_second(f, s, i, t_in, t_out)
        else:
            return s - t0*f[i]*np.log(t_out/t_in)
    else:
        return s

###VARIABLES###
t0 = temperature(25, 'k')
p0 = pressure(1.01325, 'pa')
lw = (cp('H', 'P', pressure(1.01325, 'pa'), 'Q', 1, 'water')-cp('H', 'P', pressure(1.01325, 'pa'), 'Q', 0, 'water')) / 1e3
r_gases = cp('GAS_CONSTANT', 'T', temperature(25, 'k'), 'P', pressure(1.01325, 'pa'), 'air') / 1e3
cod = 30
m_sc = mass_flow(500, 'kg')
ex_sc = 5297

prop_cp = json.load(open('/home/luismazini/Documents/phd/oxycombustion/prop_cp.json', 'r'))

class ThermoPoint:
    def __init__(self, s1:str, p1:float, s2:str, p2:float, stm:str) -> None:
        '''
        Class that describes a thermodynamic point

        :param str s1: first thermodynamic property. For instance, 'T' for temperature, 'P' for pressure.
        :param float p1: the value of the first thermodynamic property s1.
        :param str s2: second thermodynamic property. For instance, 'T' for temperature, 'P' for pressure.
        :param float p2: the value of the second thermodynamic property s2.
        :param str stm: name of the fluid (string).
        '''
        self.fld = stm #fluid
        self.t = cp('T', s1, p1, s2, p2, stm)  #temperature in K
        self.p = cp('P', s1, p1, s2, p2, stm)  #pressure in Pa
        self.h = cp('H', s1, p1, s2, p2, stm)  #enthalpy in J/kg
        self.s = cp('S', s1, p1, s2, p2, stm)  #entropy in J/kgK
        self.d = cp('D', s1, p1, s2, p2, stm)  #density in kg/m3
        self.mm = cp('M', s1, p1, s2, p2, stm) #molar mass in kg/mol

    def get_molar(self, s:str) -> float:
        '''
        Converts the enthalpy or entropy in molar basis.

        :param str s: 'h' for enthalpy or 's' for entropy.
        :return: the enthalpy or entropy in J/mol or J/molK, respectively.
        '''
        if s.lower() == 'h': #enthalpy
            return self.h / self.mm
        elif s.lower() == 's': #entropy
            return self.s / self.mm
        else:
            return 0

    def exergy(self, m:float) -> float:
        '''
        Calculates the exergy in kW

        :param float m: mass flow in kg/s.
        :return: the exergy in kW.
        '''
        return m*((self.h-cp('H', 'T', t0, 'P', p0, self.fld))/1e3 - t0*(self.s-cp('S', 'T', t0, 'P', p0, self.fld))/1e3 + prop_cp[self.fld]['chem_x'])

    def __str__(self) -> str:
        return 'Thermodynamic state:\n\n' \
            '########################### \n\n' \
            'fluid: ' + self.fld.upper() + '\n' \
            'pressure: ' + str(round(pressure(self.p, 'bar'), 2)) + ' bar \n' \
            'temperature: ' + str(round(temperature(self.t, 'c'), 2)) + ' C \n' \
            'quality: ' + str(round(self.q, 2)) + '\n' \
            'enthalpy: ' + str(round(self.h / 1000, 2)) + ' kJ/kg \n' \
            'entropy: ' + str(round(self.s / 1000, 2)) + ' kJ/kgK \n'

class FlueGases:
    def __init__(self, p_flue: float, t_flue: float, nflue: float, y_n2: float, y_co2: float, y_h2o: float) -> None:
        '''
        Class that describes the gases from the combustion chamber, which undergo the dewatering and the compression subsystems.

        :param float p_flue: flue gases pressure in Pa.
        :param float t_flue: flue gases temperature in K.
        :param float nflue: flue gases molar flow in mol/s.
        :param float y_n2: nitrogen molar fracion in the flue gases.
        :param float y_co2: carbon dioxide molar fracion in the flue gases.
        :param float y_h2O: water molar fracion in the flue gases.
        '''
        
        self.p_flue = p_flue
        self.t_flue = t_flue
        self.nflue = nflue
        self.y_n2 = y_n2
        self.y_co2 = y_co2
        self.y_h2o = y_h2o

        if p_flue > 7.38e6: #the flue gases pressure is above the carbon dioxide critical pressure
        #Therefore, there is only N2 in the gaseous phase.    
            pout_co2 = p_flue
            pout_n2 = p_flue
        else: #both CO2 and N2 are in the gaseous phase.
            pout_co2 = p_flue * self.y_co2
            pout_n2 = p_flue * self.y_n2

        if y_n2 == 0: #there's no N2 in the flue gases
            self.pt_n2 = None
        else: #there's some N2 in the flue gases 
            self.pt_n2 = ThermoPoint('T', t_flue, 'P', pout_n2, 'N2') #initiate N2 thermodynamic properties

        self.pt_co2 = ThermoPoint('T', t_flue, 'P', pout_co2, 'CO2') #initiate CO2 thermodynamic properties

        if y_h2o == 0: #there's no H2O in the flue gases
            self.pt_h2o = None
        else:
            self.pt_h2o = ThermoPoint('T', t_flue, 'P', p_flue * y_h2o, 'H2O') #initiate H2O thermodynamic properties
        #>> calculate the molar mass of the flue gases
        self.mm = y_n2*prop_cp['n2']['molar_mass'] + y_co2*prop_cp['co2']['molar_mass'] + y_h2o*prop_cp['water']['molar_mass']
        #<<
        self.mflue = nflue * self.MM #flue gases mass flow in kg/s.

    def dewatering(self) -> list:
        '''
        Remove the water from the flue gases stream and calculates the exergy content of the dewatering's heat exchange.

        :return: a list of two elements. The first is an instance of the FlueGases class, defining the flue gases stream without water. The second one is the exergy content of the heat exchange in the process of cooling water (MW).
        '''
        ntotal = self.nflue * (1-self.y_h2o) #molar flow of flue gases without water in kg/s.
        p_flue = self.p_flue * (1-self.y_h2o) #new pressure of the fluegases without the influence of the water partial pressure.
        p_water = self.p_flue * self.y_h2o #water partial pressure in Pa.
        t_water = cp('T', 'P', p_water, 'Q', 1, 'H2O') #saturated temperature at Pwater pressure in K.
        t_final = t_water - 5 #to warrantee that all the water will condensate in the cyclone.

        t_q = np.linspace(t_final, self.t_flue, 20, True) #list of temperatures from Tfinal to fluegases temperature

        #>> exergy content of cooling the fluegases: water part
        t_q_h2o = np.linspace(t_water + 5, self.t_flue, 20, True) 
        cp_h2o = list(map(lambda t: cp('C', 'T', t, 'P', p_water, 'H2O') / 1e3, t_q_h2o))
        cof_h2o = list(np.polyfit(t_q_h2o, cp_h2o, 1))
        q_ex = (int_first(cof_h2o, 0, 0, self.t_flue, t_water) 
        + int_second(cof_h2o, 0, 0, self.t_flue, t_water) 
        + (cp('H', 'P', p_water, 'Q', 0, 'water')-cp('H', 'P', p_water, 'Q', 1, 'H2O')) / 1e3 * (1-t0/t_water)
        + int_first(cof_h2o, 0, 0, t_water, t_final) 
        + int_second(cof_h2o, 0, 0, t_water, t_final)) * self.nflue * self.y_h2o * prop_cp['water']['molar_mass']
        #<<
        if self.y_n2 == 0: #there's no N2 in the flue gases
            y_n2 = 0
        else: #there's N2 in the flue gases
            #>> exergy content of cooling the fluegases: nitrogen part
            y_n2 = self.y_n2 * self.nflue / ntotal 
            p_n2 = self.p_flue * self.y_n2
            cp_n2 = list(map(lambda t: cp('C', 'T', t, 'P', p_n2, 'N2') / 1e3, t_q))
            cof_n2 = list(np.polyfit(t_q, cp_n2, 1))
            q_ex += (int_first(cof_n2, 0, 0, self.t_flue, t_final) 
            + int_second(cof_n2, 0, 0, self.t_flue, t_final)) * self.nflue * self.y_n2 * prop_cp['n2']['molar_mass'] 
            #<<

        #>> exergy content of cooling the fluegases: carbon dioxide part
        y_co2 = self.y_co2 * self.nflue / ntotal
        p_co2 = self.p_flue * self.y_co2
        cp_co2 = list(map(lambda t: cp('C', 'T', t, 'P', p_co2, 'CO2') / 1e3, t_q))
        cof_co2 = list(np.polyfit(t_q, cp_co2, 1))
        q_ex += (int_first(cof_co2, 0, 0, self.t_flue, t_final) 
        + int_second(cof_co2, 0, 0, self.t_flue, t_final)) * self.nflue * self.y_co2 * prop_cp['co2']['molar_mass'] 
        #<<
        return [FlueGases(p_flue, t_final, ntotal, y_n2, y_co2, 0), q_ex / 1e3]

    def compressor(self, p_out: float, eta: float):
        '''
        Defines a FluGases stream after the compression to a Pout pressure.

        :param float p_out: pressure after compression in Pa.
        :param float eta: compression efficiency.
        :return: an instance of the FlueGases class, defining the flue gases stream after compression.
        '''
        if p_out > 7.38e6: #the flue gases pressure is above the carbon dioxide critical pressure
            pout_co2 = p_out
        else: #CO2 is in the gaseous phase.
            pout_co2 = p_out * self.y_co2

        pt_co2 = ThermoPoint('P', pout_co2, 'S', self.pt_co2.s, 'CO2') # ThermoPoint that represents the temperature of the flue gases

        if eta < 1: #the compressor or pump is not ideal
            h_real = (pt_co2.h + self.pt_co2.h*(eta-1)) / eta
            t_real = cp('T', 'P', pout_co2, 'H', h_real, 'CO2')
        else: #the compressor or pump is ideal
            t_real = pt_co2.t
        return FlueGases(pout_co2, t_real, self.nflue, self.y_n2, self.y_co2, 0)
    
    def get_h(self) -> float:
        '''
        Get the enthalpy of the flue gases in W.

        :return: the enthalpy of the flue gases in W.
        '''
        h = 0 #initiate the enthalpy variable
        if self.y_h2o != 0: #there's water in the flue gases
            h += self.pt_h2o.h * self.y_h2o * prop_cp['water']['molar_mass'] / self.mm * self.mflue

        if self.y_n2 != 0: #there's nitrogen in the flue gases
            h += self.pt_n2.h * self.y_n2 * prop_cp['n2']['molar_mass'] / self.mm * self.mflue

        h += self.pt_co2.h * self.y_co2 * prop_cp['co2']['molar_mass'] / self.mm * self.mflue
        return h

    def get_cp(self, T) -> float:
        '''
        Get the enthalpy of the flue gases in W.

        :return: the enthalpy of the flue gases in W.
        '''
        c = 0
        if self.y_h2o != 0:
            c += cp('C', 'T', T, 'P', self.pt_h2o.p, 'water') * self.y_h2o * prop_cp['water']['molar_mass'] / self.mm

        if self.y_n2 != 0:
            c += cp('C', 'T', T, 'P', self.pt_n2.p, 'N2') * self.y_n2 * prop_cp['n2']['molar_mass'] / self.mm 

        c += cp('C', 'T', T, 'P', self.pt_co2.p, 'CO2') * self.y_co2 * prop_cp['co2']['molar_mass'] / self.mm
        return c
    
    def work_or_heat(self, flue) -> list:
        '''
        Calculates the work or heat and the heat exergy content.

        :param FlueGases flue: FlueGases instance.
        :return: a list of two elements. The first is the work or heat in kW. The second is the heat exergy content in kW.
        '''
        q_ex = 0
        t_q = np.linspace(self.t_flue, flue.t_flue, 20, True)
        
        if self.y_n2 != 0:
            cp_n2 = list(map(lambda t: cp('C', 'T', t, 'P', flue.pt_n2.p, 'N2'), t_q))
            cof_n2 = list(np.polyfit(t_q, cp_n2, 1))
            q_ex += (int_first(cof_n2, 0, 0, flue.t_flue, self.t_flue) 
            + int_second(cof_n2, 0, 0,  flue.t_flue, self.t_flue)) * self.nflue * self.y_n2 * prop_cp['n2']['molar_mass']

        cp_co2 = list(map(lambda t: cp('C', 'T', t, 'P', flue.pt_co2.p, 'CO2'), t_q))
        cof_co2 = list(np.polyfit(t_q, cp_co2, 1))
        q_ex += (int_first(cof_co2, 0, 0, flue.t_flue, self.t_flue) 
        + int_second(cof_co2, 0, 0, flue.t_flue, self.t_flue)) * self.nflue * self.y_co2 * prop_cp['co2']['molar_mass'] 
        
        w = self.get_h() - flue.get_h()
        return [w / 1e6, q_ex / 1e6]

class BioGas:
    def __init__(self, yCH4: float, yCO2: float, yH2S: float = 0, yH2O: float = 0, m: float = 1) -> None:
        '''
        Class that describes the Biogas

        :param float yCH4: molar fraction of methane in the stream.
        :param float yCO2: molar fraction of carbon dioxide in the stream.
        :param float yH2S: molar fraction of H2S in the stream.
        :param float yH2O: molar fraction of water in the stream.
        :param float m: mass flow in kg/s
        '''
        self.m = m
        #>> molar mass in kg/mol
        self.MM = yCH4*prop_cp['methane']['molar_mass'] + yCO2*prop_cp['co2']['molar_mass'] + yH2S*prop_cp['H2S']['molar_mass'] + yH2O*prop_cp['water']['molar_mass']
        #<<
        self.n = m / self.MM #mols in the biogas
        self.xCH4 = yCH4 * prop_cp['methane']['molar_mass'] / self.MM   #mass fraction of methane
        self.yCH4 = yCH4 #molar fraction
        self.xCO2 = yCO2 * prop_cp['co2']['molar_mass'] / self.MM #mass fraction of CO2
        self.yCO2 = yCO2 #molar fraction
        self.xH2S = yH2S * prop_cp['H2S']['molar_mass'] / self.MM #mass fraction of H2S 
        self.yH2S = yH2S #molar fraction
        self.xH2O = yH2O * prop_cp['water']['molar_mass'] / self.MM #mass fraction of H2O
        self.yH2O = yH2O #molar fraction

    def mols_atom(self, cp:str) -> float:
        '''
        Calculates the mols of carbon or hidrogen in biogas' methane.

        :param str cp: 'c' for carbon and 'h' for hydrogen.
        :return: mols of carbon or hydrogen in biogas' methane.
        '''
        if cp.lower() == 'c': #carbon
            return self.n * self.yCH4
        elif cp.lower() == 'h': #hydrogen
            return self.n * self.yCH4 * 4
        else:
            return 0

    def mols_molecule(self, cp:str) -> float:
        '''
        Calculates the mols of CH4, CO2, H2S, or H2O in biogas

        :param str cp: 'ch4' for methane, 'co2' for carbon dioxide, 'h2s' for hydrogen sulfuric, and 'h2o for water.
        :return: mols of CH4, CO2, H2S, or H2O in biogas
        '''
        if cp.lower() == 'ch4': #methane
            return self.n * self.yCH4
        elif cp.lower() == 'co2': #carbon dioxide
            return self.n * self.yCO2
        if cp.lower() == 'h2s': #hydrogen sulfuric
            return self.n * self.yH2S
        elif cp.lower() == 'h2o': #water
            return self.n * self.yH2O
        else:
            return 0

    def exergy(self, T: float=t0, P: float=p0) -> float:
        '''
        Calculates the exergy in kW

        :param float T: biogas temperature in K
        :param float P: biogas pressure in Pa
        :return: the exergy in kW.
        '''
        delta_ch4 = self.xCH4 * ((cp('H', 'T', T, 'P', P, 'methane')-cp('H', 'T', T, 'P', P, 'methane'))/1e3 - t0*(cp('S', 'T', T, 'P', P, 'methane')-cp('S', 'T', T, 'P', P, 'methane'))/1e3)
        delta_co2 = self.xCO2 * ((cp('H', 'T', T, 'P', P, 'co2')-cp('H', 'T', T, 'P', P, 'co2'))/1e3 - t0*(cp('S', 'T', T, 'P', P, 'co2')-cp('S', 'T', T, 'P', P, 'co2'))/1e3)
        delta_h2s = self.xH2S * ((cp('H', 'T', T, 'P', P, 'H2S')-cp('H', 'T', T, 'P', P, 'H2S'))/1e3 - t0*(cp('S', 'T', T, 'P', P, 'H2S')-cp('S', 'T', T, 'P', P, 'H2S'))/1e3)
        delta_h2o = self.xH2O * ((cp('H', 'T', T, 'P', P, 'water')-cp('H', 'T', T, 'P', P, 'water'))/1e3 - t0*(cp('S', 'T', T, 'P', P, 'water')-cp('S', 'T', T, 'P', P, 'water'))/1e3)
        chem_ex = self.xCH4*prop_cp['methane']['chem_x'] 
        + self.xCO2*prop_cp['co2']['chem_x'] 
        + self.xH2S*prop_cp['H2S']['chem_x'] 
        + self.xH2O*prop_cp['water']['chem_x']
        return self.m * (delta_ch4+delta_co2+delta_h2s+delta_h2o+chem_ex)

class LignoCel:
    def __init__(
        self, xashes: float, xwater: float, xfiber: float, xcell: float, 
        xhemi: float, xlign: float, xsucrose: float, m: float=1) -> None:
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
            'dry': m * (1-xwater)
        }
        self.mm = { #lignocellulosic material's molar mass in a wet and dry basis
            'wet': 1 / (xfiber*(xcell/prop_cp['cellulose']['molar_mass'] 
                + xhemi/prop_cp['hemicellulose']['molar_mass'] 
                + xlign/prop_cp['lignin']['molar_mass']) 
                + xsucrose/prop_cp['sucrose']['molar_mass'] 
                + xashes/prop_cp['sodium_sulfide']['molar_mass'] 
                + xwater / prop_cp['water']['molar_mass']),
            'dry':1 / (xfibern*(xcell/prop_cp['cellulose']['molar_mass'] 
                + xhemi/prop_cp['hemicellulose']['molar_mass']
                + xlign/prop_cp['lignin']['molar_mass']) 
                + xsucrosen/prop_cp['sucrose']['molar_mass'] 
                + xashesn/prop_cp['sodium_sulfide']['molar_mass'])
        }
        self.n = { #number of mols in the material in a wet and dry basis
            'wet': m / self.mm['wet'], 
            'dry': m * (1-xwater) / self.mm['dry']
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
            'wet':  xsucrose * prop_cp['sucrose']['molar_mass'] / self.mm['wet'],
            'dry': xsucrosen * prop_cp['sucrose']['molar_mass'] / self.mm['dry']
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
            'wet': self.xcell['wet'] * prop_cp['cellulose']['molar_mass'] / self.mm['wet'],
            'dry': self.xcell['dry'] * prop_cp['cellulose']['molar_mass'] / self.mm['dry']
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
            'wet': self.xhemi['wet'] * prop_cp['hemicellulose']['molar_mass'] / self.mm['wet'],
            'dry': self.xhemi['dry'] * prop_cp['hemicellulose']['molar_mass'] / self.mm['dry']
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
        self.ylign = { #molar fraction of lignin in dry and wet basis
            'wet': self.xlign['wet'] * prop_cp['lignin']['molar_mass'] / self.mm['wet'],
            'dry': self.xlign['dry'] * prop_cp['lignin']['molar_mass'] / self.mm['dry']
        }
        self.complign = { #composition of lignin
            'C': 10,
            'H': 11.5,
            'O': 3.9
        }

    def enthalpy_f(self, tp: str) -> float:
        '''
        Calculate the Enthalpy of formation of an lignocellulosic material in dry basis.

        :param str tp: use "spec_mol" for specific enthalpy of formation in kJ/mol; "spec_mass" for enthalpy of formation
        in kJ/kg; and "full" for kJ.
        :return:  Enthalpy of formation of an lignocellulosic material in dry basis. (kJ/mol, kJ/kg, or kJ).
        '''
        h = self.ycell['dry']*prop_cp['cellulose']['enthalpy_f'] 
        + self.yhemi['dry']*prop_cp['hemicellulose']['enthalpy_f'] 
        + self.ylign['dry']*prop_cp['lignin']['enthalpy_f']

        if tp.lower() == 'spec_mol':
            pass
        elif tp.lower() == 'spec_mass':
            h *= self.n['dry'] / self.m['dry']
        elif tp.lower() == 'full':
            h *= self.n['dry']
        else:
            h = None
        return h

    def mols_atom(self, at: str) -> float:
        '''
        Get the mols of a given element in the lignocel's material

        :param str at: 'C' for carbon, 'H' for hydrogen, and 'O' for oxygen.
        :return: mols of carbon, hydrogen or oxygen within an lignocel's material.
        '''
        return self.n['dry'] * (self.ycell['dry']*self.compcell[at.upper()] 
        + self.yhemi['dry']*self.comphemi[at.upper()] 
        + self.ylign['dry']*self.complign[at.upper()]
        + self.ysucrose['dry']*self.compsucrose[at.upper()])

    def get_element_percentage(self, at: str, s: str) -> float:
        '''
        Get the mol fraction of a given element in the material

        :param str at: 'C' for carbon, 'H' for hydrogen, and 'O' for oxygen.
        :param str s: 'molar' for molar fraction or 'mass' for mass fraction.
        :return: mol fraction of carbon, hydrogen or oxygen within an lignocel's material.
        '''
        if s == 'molar':
            result = self.mols_atom(at) / self.n['dry']
        else:
            result = self.mols_atom(at) * prop_cp[at.lower()]['molar_mass'] / self.m['dry']

        return result

    def get_water(self, tp: str) -> float:
        '''
        Get the amount of water in the material in kg or mols.

        :param str tp: 'mol' for mols of water, and 'mass' for kg of water.
        :return: the amount of water in the material in kg or mols
        '''
        if tp.lower() == 'mol':
            return self.m['wet'] * self.xwater / prop_cp['water']['molar_mass']
        elif tp.lower() == 'mass':
            return self.m['wet'] * self.xwater

    def hhv(self) -> float:
        '''
        Calculates the HHV content of fuels in MJ/kg. source: S. A. Channiwala and P. P. Parikh, “A unified correlation for estimating HHV of solid, 
        liquid and gaseous fuels,” Fuel, vol. 81, no. 8, pp. 1051–1063, May 2002, doi: 10.1016/S0016-2361(01)00131-4.

        :return: the HHV content of the material in MJ/kg.
        '''
        c = self.get_element_percentage('C', 'mass') * 100 #c is the carbon content expressed in %
        h = self.get_element_percentage('H', 'mass') * 100 #h is the hydrogen content expressed in %
        s = 0 #s is the sulfur content expressed in %
        o = self.get_element_percentage('O', 'mass') * 100 #o is the oxygen content expressed in %
        n = 0 #n is the nitrogen content expressed in %
        a = self.xashes['dry'] * 100 #a is the ash content expressed in %
        return (0.3491*c + 1.1783*h + 0.1005*s - 0.1034*o - 0.0151*n - 0.0211*a)

    def lhv(self):
        ''' 
        Calculates the LHV content of fuels in kJ/kg. Source: W. A. Bizzo and C. G. Sánchez, “Stoichiometry of combustion and gasification reactions 
        (Estequiometria das reações de combustão e gaseificação in Portuguese),” Biomass Gasification Technology (Tecnologia Da Gaseificação de Biomassa in Portuguese), p. 430, 2010.
        
        :return:the LHV content of the material in kJ/kg.
        '''
        return (1000*self.hhv() - lw*(9*self.get_element_percentage('H', 'mass') - self.xwater)) * (1-self.xwater)

    def beta_fuel(self) -> float:
        '''
        Calculates beta index of the chemical exergy of solid fuels. Source: Szargut, J., Morris, D.R., Steward, F.R., 1987. Exergy Analysis of Thermal, Chemical,
        #and Metallurgical Processes. Hemisphere Publishing, New York.

        :return: beta index of the chemical exergy of solid fuels (admensional)
        '''
        z_h = self.get_element_percentage('H', 'mass') #z_h is the hydrogen content expressed in decimals
        z_c = self.get_element_percentage('C', 'mass') #z_c is the carbon content expressed in decimals
        z_o = self.get_element_percentage('O', 'mass') #z_o is the oxygen content expressed in decimals
        z_n = 0 #z_n is the nitrogen content expressed in decimals
        return (1.0412 + 0.216*z_h/z_c - 0.2499*z_o/z_c*(1 + 0.7884*z_h/z_c) + 0.045*z_n/z_c)/(1 - 0.3035*z_o/z_c)

    def exergy(self) -> float:
        '''
        Calculates the sum of the standart chemical exergy in kJ/kg. Source: Szargut, J., Morris, D.R., Steward, F.R., 1987. Exergy Analysis of Thermal, Chemical,
        and Metallurgical Processes. Hemisphere Publishing, New York.

        :return: standart chemical exergy in kJ/kg.
        '''
        return self.m['wet'] * ((self.lhv()+lw*self.xwater)*self.beta_fuel() + prop_cp['water']['chem_x']*self.xwater)

class CombustionChamber:
    def __init__(self, t_comb: float, p_comb: float) -> None:
        '''
        Class that describes the combustion chamber

        :param float t_comb: temperature of the combustion chamber in K.
        :param float p_comb: pressure of the combustion chamber in Pa.
        '''
        self.t_comb = t_comb
        self.p_comb = p_comb
        self.flue = None #flue gases
        self.x_h2o = None #mass fraction of water in the flue gases
        self.y_h2o = None #molar fraction of water in the flue gases
        self.x_co2 = None #mass fraction of carbon dioxide in the flue gases
        self.y_co2 = None #molar fraction of carbon dioxide in the flue gases
        self.x_n2 = None  #mass fraction of nitrogen in the flue gases
        self.y_n2 = None  #molar fraction of nitrogen in the flue gases
        self.n_o2 = None  #mols of oxygen that enters the combustion chamber
        self.y_o2 = None  #molar fraction of oxygen that enters the combustion chamber
        self.nrec = None #mols of recirculated water in mol/s
        self.mrec = None #mass flow of recirculated water in kg/s

    def make_flue_gases(self, bagasse: LignoCel, straw: LignoCel, biogas: BioGas, y_o2: float=1):
        '''
        Calculates de nitrogen, carbon dioxide and water mass fractions in the flue gases, as well as
        the quantity of oxygen and its molar fraction that enters the combustion chamber, and the molar
        and mass flow of recirculated water

        :param LignoCel bagasse: compostion of the bagasse that enters the combustion chamber
        :param LignoCel straw: compostion of the sugarcane straw that enters the combustion chamber
        :param BioGas biogas: compostion of the biogas that enters the combustion chamber
        :param float y_o2: molar fraction of oxygen that enters the combustion chamber
        :return: None.
        '''
        nf_co2 = bagasse.mols_atom('C') + straw.mols_atom('C') + biogas.mols_atom('C') #mols of CO2 formed in the combustion
        nf_h2o = (bagasse.mols_atom('H')+straw.mols_atom('H')+biogas.mols_atom('H')) / 2 #mols of H2O formed in the combustion
        n_o2 = nf_co2 + nf_h2o/2 - (bagasse.mols_atom('O')+straw.mols_atom('O')) / 2 #mols of O2 necessary for the combustion reaction.
        self.n_o2 = n_o2
        self.y_o2 = y_o2
        n_n2 = n_o2 / y_o2 * (1-y_o2) #mols of nitrogen that enters the combustion chamber with the oxygen.
        #>> variation of sensible enthalpy of CO2, H2O, and N2 in kJ/mol
        dlt_hco2 = (cp('Hmolar', 'T', self.t_comb, 'P', self.p_comb, 'CO2')-cp('Hmolar', 'T', t0, 'P', p0, 'CO2')) / 1e3
        dlt_hh2o = (cp('Hmolar', 'T', self.t_comb, 'P', self.p_comb, 'water')-cp('Hmolar', 'T', t0, 'P', p0, 'H2O')) / 1e3
        dlt_hn2 = (cp('Hmolar', 'T', self.t_comb, 'P', self.p_comb, 'N2')-cp('Hmolar', 'T', t0, 'P', p0, 'N2')) / 1e3
        #<<
        #>> mols of water that need to be recirculated
        nrec = (nf_co2*prop_cp['co2']['enthalpy_f']
        + (nf_co2+biogas.mols_molecule('co2'))*dlt_hco2 
        + nf_h2o*(prop_cp['water']['enthalpy_f']+dlt_hh2o) 
        + n_n2*dlt_hn2 
        - (bagasse.enthalpy_f('spec_mol')+straw.enthalpy_f('spec_mol')+biogas.mols_molecule('ch4')))/(((cp('Hmolar', 'Q', 0, 'T', t0, 'water')-cp('Hmolar', 'Q', 1, 'T', t0, 'water'))/1e3 - dlt_hh2o))
        #<<
        self.nrec = nrec - (bagasse.get_water('mol')+straw.get_water('mol')) #mols of recirculated water to the combustion chamber (discounting the bagasse and straw moisture content)
        self.mrec = self.nrec * prop_cp['water']['molar_mass'] #mass flow of recirculated water in kg/s
        n_flue = nrec + nf_h2o + biogas.mols_molecule('co2') + nf_co2 + n_n2 #mols of flue gases in mol/s
        #>> mass flow of flue gases in kg/s
        m_flue = (nrec+nf_h2o)*prop_cp['water']['molar_mass'] 
        + (biogas.mols_molecule('co2')+nf_co2)*prop_cp['co2']['molar_mass'] 
        + n_n2*prop_cp['n2']['molar_mass']
        #<<
        self.y_h2o = (nrec+nf_h2o) / n_flue #molar fraction of water in the flue gases
        self.x_h2o = (nrec+nf_h2o) * prop_cp['water']['molar_mass'] / m_flue #mass fraction of water in the flue gases
        self.y_co2 = (biogas.mols_molecule('co2')+nf_co2) / m_flue #molar fraction of carbon dioxide in the flue gases
        self.x_co2 = (biogas.mols_molecule('co2')+nf_co2) * prop_cp['co2']['molar_mass'] / m_flue #mass fraction of carbon dioxide in the flue gases
        self.x_n2 = n_n2 * prop_cp['n2']['molar_mass'] / m_flue #mass fraction of nitrogen in the flue gases
        self.y_n2 = n_n2 / n_flue
        self.flue = FlueGases(self.p_comb, self.t_comb, n_flue, self.y_n2, self.y_co2, self.y_h2o) #stating the flue gases of the Stream class.
        return self.flue

class Turbine:
    def __init__(self, pt_in: ThermoPoint, p_out: float, stm: str, n: float=1, massflow: float=1) -> None:
        '''
        Class that describes a turbine (Rankine or Brayton)

        :param ThermoPoint pt_in: point that describes the thermodynamic properties of the flow that enters the turine.
        :param float p_out: pressure of the outlet Turbine in Pa
        :param str stm: name of the working fluid.
        :param float n: efficiency of the turbine
        :param float massflow: massflow of the working fluid in kg/s.
        '''
        self.pt_in = pt_in #inlet thermodynamic point
        self.m = massflow 
        h_ideal = cp('H', 'P', p_out, 'S', self.pt_in.s, stm) #ideal enthalpy in J/kg
        if n < 1: #the turbine is not ideal
            h_real = n*h_ideal + self.pt_in.h*(1-n)
        else: #the turbine is ideal
            h_real = h_ideal
        self.pt_out = ThermoPoint('P', p_out, 'H', h_real, stm) #the outlet themodynamic point
        self.n = n #efficiency
    
    def work(self) -> float:
        '''
        Calculates the turbine generated power in kW

        :return: turbine generated power in kW.
        '''
        return self.m * (self.pt_in.h-self.pt_out.h) / 1e3

class PumpOrCompressor:
    def __init__(self, pt_in: ThermoPoint, p_out: float, stm: str, n: float, massflow: float=1)-> None:
        '''
        Class that describes a pump or a compressor

        :param ThermoPoint pt_in: point that describes the thermodynamic properties of the flow that enters the compressor or pump.
        :param float p_out: pressure of the outlet compressor or pump in Pa
        :param str stm: name of the working fluid.
        :param float n: efficiency of the the compressor or pump
        :param float massflow: massflow of the working fluid in kg/s.
        '''
        self.pt_in = pt_in #inlet thermodynamic point
        self.m = massflow
        h_ideal = cp('H', 'P', p_out, 'S', self.pt_in.s, stm) #ideal enthalpy in J/kg
        if n < 1: #the compressor or pump is not ideal
            h_real = (h_ideal + self.pt_in.h*(n-1)) / n
        else: #the compressor or pump is ideal
            h_real = h_ideal
        self.pt_out = ThermoPoint('P', p_out, 'H', h_real, stm) #the outlet themodynamic point
        self.n = n #efficiency

    def work(self) -> float:
        '''
        Calculates the compressor or pump consumed power in kW

        :return: compressor or pump consumed power in kW.
        '''
        return (self.pt_in.h-self.pt_out.h) / 1e3

class SingleHeater:
    def __init__(self, pt_in:ThermoPoint, pt_out:ThermoPoint, massflow:float = 1) -> None:
        '''
        Class that describes a simple heater

        :param ThermoPoint pt_in: point that describes the thermodynamic properties of the flow that enters the heater.
        :param ThermoPoint pt_out: point that describes the thermodynamic properties of the flow that leaves the heater.
        :param float massflow: massflow of the heating fluid in kg/s.
        '''
        self.pt_in = pt_in   #inlet thermodynamic point
        self.pt_out = pt_out #outlet thermodynamic point
        self.m = massflow  #massflow

    def heat(self):
        '''
        Calculates the heat power in kW

        :return: cheat power in kW.
        '''
        return(self.pt_out.h-self.pt_in.h) / 1e3

    def exergy(self):
        '''
        Calculates the exergetic content of the heat in kW.

        :return: the exergetic content of the heat in kW.
        '''
        t_q = list(range(int(self.pt_in.t - 10), int(self.pt_out.t + 10))) #list of temperatures to calculate the cp against temperature function
        cp_q = list(map(lambda x: cp('C', 'T', x, 'P', p0, self.pt_out.fld) / 1e3, t_q)) #list of cps to calculate the cp against temperature function
        cof = list(np.polyfit(t_q, cp_q, 1)) #coeficients of the polynomium that represents the cp against temperature.
        return int_first(cof, 0, 0, self.pt_in.t, self.pt_out.t) + int_second(cof, 0, 0, self.pt_in.t, self.pt_out.t)

class DoubleHeater:
    def __init__(self, pt_hot1, what_hot: int, pt_cold1: ThermoPoint, m_hot: float=1, m_cold: float=1) -> None:
        '''
        Class that describes a regenerative heater

        :param ThermoPoint or FlueGases pt_hot1: point that describes the thermodynamic properties of the flow that enters the hot part of the heater.
        :param int what_hot: 0 for ThermoPoint, 1 for FlueGases
        :param ThermoPoint pt_cold1: point that describes the thermodynamic properties of the flow that enters the cold part of the heater.
        :param int what_cold: 0 for ThermoPoint, 1 for FlueGases
        :param float mHot: heating fluid massflow of the hot part in kg/s.
        :param float mCold: heating fluid massflow of the cold part in kg/s.
        '''
        self.pt_hot1 = pt_hot1
        self.what_hot = what_hot
        self.pt_hot2 = None #point that describes the thermodynamic properties of the flow that leaves the hot part of the heater.
        self.pt_cold1 = pt_cold1
        #self.whatCold = whatCold
        self.pt_cold2 = None #point that describes the thermodynamic properties of the flow that leaves the cold part of the heater.
        self.m_hot = m_hot
        self.m_cold = m_cold

        #>> list of the cold part temperatures to calculate the polynomium that will describe the cp(T) function of the cold part.
        list_temperature_cold = list(range(int(pt_cold1.t - 10), int(pt_cold1.t + 100), 9))
        #<<
        #>> list of the cold part cps to calculate the polynomium that will describe the cp(T) function of the cold part.
        list_cpcold = list(map(lambda x: cp('C', 'T', x, 'P', pt_cold1.p, pt_cold1.fld), list_temperature_cold))
        #<<
        self.cp_tcold = list(np.polyfit(list_temperature_cold, list_cpcold, 1)) #cold cp(T) -> coeficients
        #>> Heat capacity of the cold part. It will be used to compare wich one is bigger: the hot or cold one.
        self.c_cold = m_cold * sum(list_cpcold) / len(list_cpcold)
        #<<
        if what_hot == 0:
            #>> list of the hot part temperatures to calculate the polynomium that will describe the cp(T) function of the hot part.
            list_temperature_hot = list(range(int(pt_hot1.t - 100), int(pt_hot1.t + 10), 9))
            #<<
            #>> list of the hot part cps to calculate the polynomium that will describe the cp(T) function of the hot part.
            list_cphot = list(map(lambda x: cp('C', 'T', x, 'P', pt_hot1.p, pt_hot1.fld), list_temperature_hot))
            #<<
            self.cp_thot = list(np.polyfit(list_temperature_hot, list_cphot, 1))    #hot cp(T) -> coeficients
            #>> Heat capacity of the hot part. It will be used to compare wich one is bigger: the hot or cold one.
            self.c_hot = m_hot * sum(list_cphot) / len(list_cphot)
            #<<
        else:
            #>> list of the hot part temperatures to calculate the polynomium that will describe the cp(T) function of the hot part.
            list_temperature_hot = list(range(int(pt_hot1.t_flue - 100), int(pt_hot1.t_flue + 10), 9))
            #<<
            #>> list of the hot part cps to calculate the polynomium that will describe the cp(T) function of the hot part.
            list_cphot = list(map(lambda x: pt_hot1.get_cp(x), list_temperature_hot))
            #<<
            self.cp_thot = list(np.polyfit(list_temperature_hot, list_cphot, 1))    #hot cp(T) -> coeficients
            #>> Heat capacity of the hot part. It will be used to compare wich one is bigger: the hot or cold one.
            self.c_hot = m_hot * sum(list_cphot) / len(list_cphot)
            #<<

    def ua_inf(self) -> None:
        '''
        Calculates and make the themodynamic points that leaves both the hot and cold part of the regenerative heater, considering 
        an infinite area for heat exchanging.

        :return: None
        '''
        #>> The hot part heat capacity is bigger then the cold one. It means that the cold part experience more temperature variation than
        # the hot part. Because of that, pt_cold2.t (outlet temperature of the cold part) is going to be equal to the pt_hot1.t (intlet temperature of the hot part)
        if self.c_hot > self.c_cold: 
            if self.what_hot == 0:
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', self.pt_hot1.t, self.pt_cold1.fld)
                self.pt_hot2 = ThermoPoint('P', self.pt_hot1.p, 'T', newton(lambda t: 
                - self.m_hot*integral(self.cp_thot, 0, 0, self.pt_hot1.t, t)
                - self.m_cold*integral(self.cp_tcold, 0, 0, self.pt_cold1.t, self.pt_cold2.t), self.pt_cold1.t, fprime=lambda t: 
                - self.m_hot*cp_function(self.cp_thot, 0, 0, self.pt_hot1.t, t) 
                - self.m_cold*cp_function(self.cp_tcold, 0, 0, self.pt_cold1.t, self.pt_cold2.t)), self.pt_hot1.fld)
            else:
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', self.pt_hot1.t_flue, self.pt_cold1.fld)
                self.pt_hot2 = FlueGases(self.pt_hot1.p_flue, newton(lambda t:
                - self.m_hot*integral(self.cp_thot, 0, 0, self.pt_hot1.t_flue, t)
                - self.m_cold*integral(self.cp_tcold, 0, 0, self.pt_cold1.t, self.pt_cold2.t), self.pt_cold1.t, fprime=lambda t:
                - self.m_hot*cp_function(self.cp_thot, 0, 0, self.pt_hot1.t_flue, t)
                - self.m_cold*cp_function(self.cp_tcold, 0, 0, self.pt_cold1.t, self.pt_cold2.t)), self.pt_hot1.nflue, self.pt_hot1.y_n2, self.pt_hot1.y_co2, self.pt_hot1.y_h2o)
        #<<
        #>> The cold part heat capacity is bigger then the hot one. It means that the hot part experience more temperature variation than
        # the cold part. Because of that, pt_hot2.t (outlet temperature of the hot part) is going to be equal to the pt_cold1.t (intlet temperature of the cold part)
        elif self.c_hot < self.c_cold:
            if self.what_hot == 0:
                self.pt_hot2 = ThermoPoint('P', self.pt_hot1.p, 'T', self.pt_cold1.t, self.pt_hot1.fld)
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', newton(lambda t: 
                - self.m_hot*integral(self.cp_thot, 0, 0, self.pt_hot1.t, self.pt_hot2.t) 
                - self.m_cold*integral(self.cp_tcold, 0, 0, self.pt_cold1.t, t), self.pt_hot1.t, fprime=lambda t: 
                - self.m_hot*cp_function(self.cp_thot, 0, 0, self.pt_hot1.t, self.pt_hot2.t) 
                - self.m_cold*cp_function(self.cp_tcold, 0, 0, self.pt_cold1.t, t)), self.pt_cold1.fld)
            else:
                self.pt_hot2 = FlueGases(self.pt_hot1.p_flue, self.pt_cold1.t, self.pt_hot1.nflue, self.pt_hot1.y_n2, self.pt_hot1.y_co2, self.pt_hot1.y_h2o)
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', newton(lambda t: 
                - self.m_hot*integral(self.cp_thot, 0, 0, self.pt_hot1.t_flue, self.pt_hot2.t_flue) 
                - self.m_cold*integral(self.cp_tcold, 0, 0, self.pt_cold1.t, t), self.pt_hot1.t_flue, fprime=lambda t: 
                - self.m_hot*cp_function(self.cp_thot, 0, 0, self.pt_hot1.t_flue, self.pt_hot2.t_flue) 
                - self.m_cold*cp_function(self.cp_tcold, 0, 0, self.pt_cold1.t, t)), self.pt_cold1.fld)
        #<<
        #>> The hot part heat capacity is equal to the cold one. It means that the cold and hot parts experience the same temperature variation. 
        # Because of that, pt_cold2.t (outlet temperature of the cold part) is going to be equal to the pt_hot1.t (intlet temperature of the hot part),
        # and pt_hot2.t (outlet temperature of the hot part) is going to be equal to the pt_cold1.t (intlet temperature of the cold part).
        else:
            if self.what_hot == 0:
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', self.pt_hot1.t, self.pt_cold1.fld)
                self.pt_hot2 = ThermoPoint('P', self.pt_hot1.p, 'T', self.pt_cold1.t, self.pt_hot1.fld)
            else:
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', self.pt_hot1.t_flue, self.pt_cold1.fld)
                self.pt_hot2 = FlueGases(self.pt_hot1.p_flue, self.pt_cold1.t, self.pt_hot1.nflue, self.pt_hot1.y_n2, self.pt_hot1.y_co2, self.pt_hot1.y_h2o)
        #<<
        return None

    def get_pt(self, t_ch:float, phase:str) -> None:
        '''
        Calculates and make the themodynamic points that leaves both the hot and cold part of the regenerative heater, considering 
        an infinite area for heat exchanging, but restricting the outlet temperature to Tch in the cold or hot phases.

        :param float t_ch: restricted temperature in K.
        :param str phase: phase that you want to restringe. If phase = 'cold', it will restringe the cold part. If phase = 'hot', it will restringe the hot part.
        :return: None
        '''
        if phase.lower() == 'cold':
            if self.what_hot == 0:
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', t_ch, self.pt_cold1.fld)
                self.pt_hot2 = ThermoPoint('P', self.pt_hot1.p, 'T', newton(lambda t: 
                - self.m_hot*integral(self.cp_thot, 0, 0, self.pt_hot1.t, t) 
                - self.m_cold*integral(self.cp_tcold, 0, 0, self.pt_cold1.t, t_ch), self.pt_cold1.t, fprime=lambda t: 
                - self.m_hot*cp_function(self.cp_thot, 0, 0, self.pt_hot1.t, t) 
                - self.m_cold*cp_function(self.cp_tcold, 0, 0, self.pt_cold1.t, t_ch)), self.pt_hot1.fld)
            else:
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', t_ch, self.pt_cold1.fld)
                self.pt_hot2 = FlueGases(self.pt_hot1.p_flue, newton(lambda t:
                - self.m_hot*integral(self.cp_thot, 0, 0, self.pt_hot1.t_flue, t)
                - self.m_cold*integral(self.cp_tcold, 0, 0, self.pt_cold1.t, t_ch), self.pt_cold1.t, fprime=lambda t:
                - self.m_hot*cp_function(self.cp_thot, 0, 0, self.pt_hot1.t_flue, t)
                - self.m_cold*cp_function(self.cp_tcold, 0, 0, self.pt_cold1.t, t_ch)), self.pt_hot1.nflue, self.pt_hot1.y_n2, self.pt_hot1.y_co2, self.pt_hot1.y_h2o)
        elif phase.lower() == 'hot':
            if self.what_hot == 0:
                self.pt_hot2 = ThermoPoint('P', self.pt_hot1.p, 'T', t_ch, self.pt_hot1.fld)
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', newton(lambda t: 
                - self.m_hot*integral(self.cp_thot, 0, 0, self.pt_hot1.t, t_ch) 
                - self.m_cold*integral(self.cp_tcold, 0, 0, self.pt_cold1.t, t), self.pt_hot1.t, fprime=lambda t: 
                - self.m_hot*cp_function(self.cp_thot, 0, 0, self.pt_hot1.t, t_ch) 
                - self.m_cold*cp_function(self.cp_tcold, 0, 0, self.pt_cold1.t, t)), self.pt_cold1.fld)
            else:
                self.pt_hot2 = FlueGases(self.pt_hot1.p_flue, t_ch, self.pt_hot1.nflue, self.pt_hot1.y_n2, self.pt_hot1.y_co2, self.pt_hot1.y_h2o)
                self.pt_cold2 = ThermoPoint('P', self.pt_cold1.p, 'T', newton(lambda t: 
                - self.m_hot*integral(self.cp_thot, 0, 0, self.pt_hot1.t_flue, t_ch) 
                - self.m_cold*integral(self.cp_tcold, 0, 0, self.pt_cold1.t, t), self.pt_hot1.t_flue, fprime=lambda t: 
                - self.m_hot*cp_function(self.cp_thot, 0, 0, self.pt_hot1.t_flue, t_ch) 
                - self.m_cold*cp_function(self.cp_tcold, 0, 0, self.pt_cold1.t, t)), self.pt_cold1.fld)
        return None

###PSO CLASSES###
class CompressorParticle:
    def __init__(self, p_in: float, p_out: float, n_compressor: int) -> None:
        '''
        Class that describes the particle, which is the Compressor Tandem system.

        :param float p_in: inlet pressure of the compressor in Pa.
        :param float p_out: outlet pressure of the compressor in Pa.
        :param int n_compressor: number of compressor in the Compressor tandem system. It is always >= 2.
        '''
        if n_compressor == 2:  # it means that there's only one intermediate pressure in the system
            pressure_pos = np.array([(p_in+p_out) / 2])
        else:  # there qre more than one intermediate pressure
            pressure_pos = np.linspace(p_in + 1e3, p_out, n_compressor - 1, False)

        self.pressure_position = pressure_pos
        self.pbest_position = self.pressure_position
        self.pbest_value = float('-inf') #initiate the best individual value
        self.velocity = np.array([0] * len(self.pressure_position)) #initiate the velocity element of PSO

    def __str__(self):
        print("Intermediary positions ", self.pressure_position, " Tandem Work ", self.pbest_value)
    
    def move(self) -> None:
        '''
        Function that moves the particle in the defined space
        '''
        self.pressure_position = self.pressure_position + self.velocity
 
class Tandem:
    def __init__(self, n_compressor: int, n_particles: int, p_out: float, flow: FlueGases, t_in: float) -> None:
        '''
        Class that describes the Space that the Particle will move in PSO simulation

        :param int n_compressor: number of compressor in the Compressor tandem system. It is always >= 2.
        :param int n_particles: number of particles in the PSO simulation
        :param float p_out: outlet pressure of the compressor in Pa.
        :param FlueGases flow: fluegases that is undergoing the compression process.
        :param float t_in: intlet temperature of the compressor in Pa.
        '''
        self.n_particles = n_particles
        self.p_in = flow.p_flue 
        self.p_out = p_out
        self.flow = flow
        self.t_in = t_in
        self.n_compressor = n_compressor
                
        self.particles = None             # list that saves the simulation particles
        self.heatex_comp = None           # exergy content of the tandem compressor
        self.gbest_value = float('-inf')  # initiate the global best value
        self.gbest_position = None        # initiate the global best position
    

    def fitness(self, particle: CompressorParticle) -> float:
        '''
        Function that describes the goal of the PSO simulation

        :param CompressorParticle particle: a particle of the simulation
        :return: the total power consumed by the tandem system in MW
        '''
        def total_work(i: int) -> float:
            '''
            Function that describes the goal of the PSO simulation for each compressor

            :param int i: variable counter for the compressors.
            :return: the total power consumed by the compressor in MW
            '''
            if i == 0:                                         # the first compressor
                if particle.pressure_position[i] < self.p_in:  # it means that the initial presure
                    # of the first compressor is smaller than the inlet pressure of the system.
                    particle.pressure_position[i] = self.p_in + 1e4

                p_in = self.p_in
                p_out = particle.pressure_position[i]
            elif i == self.n_compressor-1:  # the last compressor 
                if particle.pressure_position[-1] > self.p_out:  # it means that the final presure
                    # of the last compressor is bigger than the outlet pressure of the system.
                    particle.pressure_position[-1] = self.p_out - 1e4

                p_in = particle.pressure_position[-1]
                p_out = self.p_out
            else:  # some intermediary compressor
                if particle.pressure_position[i-1] > particle.pressure_position[i]:
                    aux = particle.pressure_position[i]
                    particle.pressure_position[i] = particle.pressure_position[i-1]
                    particle.pressure_position[i-1] = aux

                p_in = particle.pressure_position[i-1]
                p_out = particle.pressure_position[i]
                
            pt = FlueGases(p_in, self.t_in, self.flow.nflue, self.flow.y_n2, self.flow.y_co2, self.flow.y_h2o)
            return pt.work_or_heat(pt.compressor(p_out, 1))

        return np.sum(list(map(total_work, range(self.n_compressor))), axis=0)

    def set_pbest(self) -> None:
        '''
        Function that updates the particles' best position

        :return: None
        '''
        def return_pbest(particle: CompressorParticle) -> CompressorParticle:
            '''
            Function that updates the particle's best position

            :param CompressorParticle particle: a particle of the simulation
            :return: a CompressorParticle with its best position by now.
            '''
            fitness_cadidate = self.fitness(particle)[0]

            if abs(particle.pbest_value) > abs(fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.pressure_position

            return particle
        
        self.particles = list(map(lambda p: return_pbest(p), self.particles))
        return None
            
    def set_gbest(self) -> None:
        '''
        Function that updates the global best position

        :return: None
        '''
        def return_gbest(n: int) -> None:
            '''
            Function that updates the global best position

            :param int n: variable counter for the list of particles.
            :return: None
            '''
            fitness = self.fitness(self.particles[n])
            best_fitness_cadidate = fitness[0]
            ex_heat = fitness[1]  # exergy content of the heat exchange in the compression tandem system

            if abs(self.gbest_value) > abs(best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = self.particles[n].pressure_position
                self.heatex_comp = ex_heat

            if n < len(self.particles) - 1:
                n += 1
                return return_gbest(n)
            else:
                return None

        return_gbest(0)
        return None

    def move_particles(self, w: float, c1: float, c2: float) -> None:
        '''
        Function that moves all the particles
        
        :param float w: the inertia term.
        :param float c1: the cognitive term.
        :param float c2: the social term.
        :return: None
        '''
        def move_particle(n) -> None:
            '''
            Function that moves the n-th particle

            :param int n: variable counter for the list of particles.
            :return: None
            '''
            new_velocity = w*self.particles[n].velocity 
            + c1*random.random()*(self.particles[n].pbest_position-self.particles[n].pressure_position) 
            + c2*random.random()*(self.gbest_position-self.particles[n].pressure_position)

            self.particles[n].velocity = new_velocity
            self.particles[n].move()

            if n < len(self.particles) - 1:
                n += 1
                return move_particle(n)
            else:
                return None

        move_particle(0)
        return None

class Particle():
    def __init__(self, rf: float, lim_cf: float) -> None:
        '''
        Class that describes the particle of the PSO method

        :param float rf: recovery factor (oxygen molar flow that enters the combustion chamber / oxygen molar flow that enters the membrane separation process)
        :param float lim_cf: the max value of the concentration factor.
        '''
        #>> initiate the Concentration Factor (oxygen molar fraction that enters the combustion chamber/oxygen molar fraction in the atmospheric air)
        self.ConcentrationFactor = random.random() * lim_cf
        #<< 
        self.RecoveryFactor = rf
        self.pbest_CF = self.ConcentrationFactor #initiate the best coordinates for the best individual value
        self.pbest_value = float('inf') #initiate the best individual value
        self.velocity = 0 #initiate the velocity element of PSO

    def __str__(self):
        print("Recovery Factor ", self.RecoveryFactor, " Concentration Factor ", self.pbest_CF)
    
    def move(self) -> None:
        '''
        Function that moves the particle in the defined space
        '''
        self.ConcentrationFactor = self.ConcentrationFactor + self.velocity
 
class Space():
    def __init__(self, target_error: float, n_particles: int, n: float, lim_cf: float, y_o2in: float) -> None:
        '''
        Class that describes the Space that the Particle will move in PSO simulation

        :param float target_error: permissible error between the target and the fitfunction value.
        :param int n_particles: number of particles in the PSO simulation
        :param float n: fraction of the minimum separation work
        :param float lim_cf: the max value of the concentration factor.
        :param float y_o2in: the initial concentration of oxygen in the atmospheric air
        '''
        self.n = n
        self.target = 0 #target of the fitfunction
        self.target_error = target_error
        self.n_particles = n_particles
        self.lim_cf = lim_cf
        self.particles = [] #list that saves the simulation particles
        self.gbest_value = float('inf') #initiate the global best value
        self.gbest_position = random.random() * lim_cf #initiate the global best position
        self.y_o2in = y_o2in
   
    def fitness(self, particle: Particle) -> float:
        '''
        Function that describes the goal of the PSO simulation

        :param Particle particle: a particle of the simulation
        '''
        y_n2in = 1 - self.y_o2in
        y_o2out = self.y_o2in * particle.ConcentrationFactor # oxygen molar fraction that enters the combustion chamber
        y_n2out = 1 - y_o2out #nitrogen molar fraction that enters the combustion chamber
        beta = particle.RecoveryFactor * self.y_o2in / y_o2out #outlet molar flow / inlet molar flow
        y_o2_2 = (self.y_o2in - beta*y_o2out) / (1-beta) #oxygen molar fraction of the non-interesting outlet
        y_n2_2 = 1 - y_o2_2 #nitrogen molar fraction of the non-interesting outlet
        #>> F(ConcentrationFactor) = 0. If the function value is indeterminate, the function returns infinite
        try:
            fCF = (particle.ConcentrationFactor ** (-1*self.y_o2in)) * y_n2in * ((y_o2out/y_n2out)**(-y_o2out*beta)) * y_n2out**(-beta) * ((y_o2_2/y_n2_2)**(y_o2_2 * (beta-1))) * y_n2_2**(beta-1) - np.exp(self.n * (self.y_o2in*np.log(self.y_o2in) + y_n2in*np.log(y_n2in)))
        except:
            fCF = float('inf')
        return fCF
        #<<

    def set_pbest(self):
        '''
        Function that finds the best position for a particle

        :return: None
        '''
        def return_pbest(particle: Particle) -> Particle:
            '''
            Function that finds the best position for a particle

            :param Particle particle: a particle of the simulation
            :return: a Particle with its best position
            '''
            try:
                fitness_cadidate = self.fitness(particle)
                if abs(particle.pbest_value) > abs(fitness_cadidate):
                    particle.pbest_value = fitness_cadidate
                    particle.pbest_CF = particle.ConcentrationFactor
            except:
                pass
            return particle
        
        self.particles = list(map(lambda p: return_pbest(p), self.particles))
        return None
            
    def set_gbest(self) -> None:
        '''
        Function that updates the global best position

        :return: None
        '''
        def return_gbest(n) -> None:
            '''
            Function that updates the global best position

            :param int n: variable counter for the list of particles.
            :return: None
            '''
            try:
                best_fitness_cadidate = self.fitness(self.particles[n])
                if abs(self.gbest_value) > abs(best_fitness_cadidate):
                    self.gbest_value = best_fitness_cadidate
                    self.gbest_position = self.particles[n].ConcentrationFactor
            except:
                pass

            if n < len(self.particles) - 1:
                n += 1
                return return_gbest(n)
            else:
                return None

        return_gbest(0)
        return None

    def move_particles(self, w: float, c1: float, c2: float) -> None:
        '''
        Function that moves all the particles
        
        :param float w: the inertia term.
        :param float c1: the cognitive term.
        :param float c2: the social term.
        :return: None
        '''
        def move_particle(n):
            '''
            Function that moves the n-th particle

            :param int n: variable counter for the list of particles.
            :return: None
            '''
            new_velocity = w*self.particles[n].velocity 
            + c1*random.random()*(self.particles[n].pbest_CF-self.particles[n].ConcentrationFactor) 
            + c2*random.random()*(self.gbest_position-self.particles[n].ConcentrationFactor)
            self.particles[n].velocity = new_velocity
            
            if (self.particles[n].ConcentrationFactor+new_velocity) <= self.lim_cf:
                self.particles[n].move()

            if n < len(self.particles) - 1:
                n += 1
                return move_particle(n)
            else:
                return None

        move_particle(0)
        return None
   
###PSO FUNCTIONS###          
def initiate_tandem(n_compressor: int, n_particles: int, p_out: float, flow: FlueGases) -> Tandem:
    '''
    Function that initiates the compression tandem system

    :param int n_compressor: number of compressor in the Compressor tandem system. It is always >= 2.
    :param int n_particles: number of particles in the PSO simulation
    :param float p_out: outlet pressure of the compressor system in Pa.
    :param FlueGases flow: fluegases that is undergoing the compression process.
    :return: the initiated compression tandem system
    '''
    search_tandem = Tandem(n_compressor, n_particles, p_out, flow, flow.t_flue)
    search_tandem.particles = list(map(lambda _: CompressorParticle(search_tandem.p_in, p_out, n_compressor), range(n_particles)))
    return search_tandem

def pso_simulation_tandem(search_space: Tandem, n_iterations: int, n: int=0, w: float=1, c1: float=0.8, c2: float=2) -> list:
    '''
    Function that describes the PSO method

    :param Tandem search_space: initiated compression tandem system.
    :param int n_iterations: method's max number of iterations.
    :param int n: iteration's variable counter.
    :param float w: the inertia term.
    :param float c1: the cognitive term.
    :param float c2: the social term.
    :return: the optimized power consumption of the compression tandem system in MW and the exergy content of heat exchange in the system in MW
    '''
    
    search_space.set_pbest()    
    search_space.set_gbest()

    if n < n_iterations:
        n += 1
        search_space.move_particles(w, c1, c2)
        return pso_simulation_tandem(search_space, n_iterations, n)
    else:
        return [search_space.gbest_value, search_space.heatex_comp]

def initiate_spaces(target_error: float, n_particles: int, n: float, rf: float, lim_cf: float, y_o2in: float=0.21) -> Space:
    '''
    Function that initiates the membrane separation Space

    :param float target_error: the target error of the simulation
    :param int n_particles: number of particles in the PSO simulation
    :param float n: fraction of the minimum separation work
    :param float rf: recovery factor (oxygen molar flow that enters the combustion chamber / oxygen molar flow that enters the membrane separation process)
    :param float lim_cf: the max value of the concentration factor.
    :return: the initiate Space
    '''
    search_space = Space(target_error, n_particles, n, lim_cf, y_o2in)
    search_space.particles = list(map(lambda _: Particle(rf, lim_cf), range(n_particles)))
    return search_space

def pso_simulation(search_space: Space, n_iterations: int, n: int, w: float=1, c1: float=0.8, c2: float=2) -> Space:
    '''
    Function that describes the PSO method

    :param Space search_space: initiated Space.
    :param int n_iterations: method's max number of iterations.
    :param int n: iteration's variable counter.
    :param float w: the inertia term.
    :param float c1: the cognitive term.
    :param float c2: the social term.
    :return: the optimized Concentration Factor
    '''
    search_space.set_pbest()    
    search_space.set_gbest()

    if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
        return search_space.gbest_position

    elif n < n_iterations:
        n += 1
        search_space.move_particles(w, c1, c2)
        return pso_simulation(search_space, n_iterations, n)
    else:
        return search_space.gbest_position