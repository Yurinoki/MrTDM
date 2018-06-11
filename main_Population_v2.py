import math
import numpy as np
import scipy.integrate as spi
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import TEIC_Astellas
import TEIC_Tsuji
import VCM_Shionogi
import VCM_Meiji
import VCM_Paediatrics
import ABK
import DAP

def get_Time(sequence):
    DateTime = sequence.Date + ' ' + sequence.Time
    DateTime2 = [datetime.strptime(x, '%Y/%m/%d %H:%M') for x in DateTime]
    delta = sequence.ID + 0.0
    delta[0] = 0
    for idx in range(1, len(delta)):
        dt = (DateTime2[idx] - DateTime2[0])
        delta[idx] = dt.days * 24 + dt.seconds / 60 / 60
    return delta

def do_Population():
    global theta
    global omega
    
    if sequence.Drug[0] == 3:
        # LZD
        # PK
        tvV1 = 22.9
        tvV2 = 24.7
        tvCLnr = 1.86
        tvCLr  = 1.44
        dCLnrdAge = -0.021
        tvQ    = 10.9
        sgPK = 0.318
        wCL = math.sqrt(0.136)
        wV1 = math.sqrt(2.02)
        wV2 = math.sqrt(0.00250)
        wQ  = math.sqrt(3.32)

        # PLT
        tvMTT = 113.0
        tvGamma = -0.187
        tvSlope = 0.00566
        wMTT = math.sqrt(0.0571)
        wGamma = math.sqrt(0.0942)
        wSlope = math.sqrt(0.224)
        sgPLT  = 0.234
        
        theta = [tvV1, tvV2, tvCLnr, tvCLr, dCLnrdAge, tvQ, tvMTT, tvGamma, tvSlope]
        omega = [wCL, wV1, wV2, wQ, wMTT, wGamma, wSlope]

        for isim in range(NSIM):
            etas = np.random.randn(len(omega)) * omega
            
            _ipred = PRED_LZD(theta, etas)
            dv["isim"]          = isim
            dv["Concentration"] = _ipred[0]
            dv["PLT"]           = _ipred[1]
            ipred_2 = pd.DataFrame(dv, columns=['isim', 'ID','Sequence','Drug','Time','Concentration','PLT'])
            if isim == 0:
                ipredAll = ipred_2
            else:
                ipredAll = pd.concat([ipredAll, ipred_2])
                
    elif sequence.Drug[0] == 1:
        # TEIC (AStellas)
        tvCL1 = 0.00498
        tvCL2 = 0.00426
        tvV1  = 10.4
        tvK12 = 0.380
        tvK21 = 0.0485
        
        wCL  = 0.221
        wV1  = 0.267
        wK21 = 0.245
        
        theta = [tvCL1, tvCL2, tvV1, tvK12, tvK21]
        omega = [wCL, wV1, wK21]

        for isim in range(NSIM):
            etas = np.random.randn(len(omega)) * omega
            
            _ipred = PRED_TEIC_Astellas(theta, etas)
            dv["isim"]          = isim
            dv["Concentration"] = _ipred[0]
            ipred_2 = pd.DataFrame(dv, columns=['isim', 'ID','Sequence','Drug','Time','Concentration'])
            if isim == 0:
                ipredAll = ipred_2
            else:
                ipredAll = pd.concat([ipredAll, ipred_2])
    
    elif sequence.Drug[0] == 2:
        # TEIC (Tsuji)
        tvCL1 = 0.466
        tvCL2 = 0.383
        tvCL3 = 0.772
        tvV11 = 57.1
        tvV12 = 0.499
        tvQ   = 0.517
        tvV2  = 34.0
        
        wCL = 0.327
        wV1 = 0.400
        rCLV1 = 0.55
        wCLV1 = rCLV1 * wCL * wV1
        
        theta = [tvCL1, tvCL2, tvCL3, tvV11, tvV12, tvQ, tvV2]
        omega = [wCL, wV1]
        
        for isim in range(NSIM):
            #etas = np.random.randn(len(omega)) * omega
            etas0 = np.random.randn(2)
            etas = [wCL * etas0[0], wCLV1 * etas0[0] + np.sqrt(wV1**2 - wCL**2 / wCL**2) * etas0[1]]
            
            _ipred = PRED_TEIC_Tsui(theta, etas)
            dv["isim"]          = isim
            dv["Concentration"] = _ipred[0]
            ipred_2 = pd.DataFrame(dv, columns=['isim', 'ID','Sequence','Drug','Time','Concentration'])
            if isim == 0:
                ipredAll = ipred_2
            else:
                ipredAll = pd.concat([ipredAll, ipred_2])
    
    elif sequence.Drug[0] == 4:
        # VCM (Shionogi)
        tvCL1 = 0.0478
        tvCL2 = 3.51
        tvVss = 60.7
        tvK12 = 0.525
        tvK21 = 0.213
        
        wCL = 0.385
        wVss = 0.254
        wK21 = 0.286

        theta = [tvCL, tvVss, tvK12, tvK21]
        omega = [wCL, wVss, wK21]

        for isim in range(NSIM):
            etas = np.random.randn(len(omega)) * omega
            
            _ipred = PRED_VCM_Shionogi(theta, etas)
            dv["isim"]          = isim
            dv["Concentration"] = _ipred[0]
            ipred_2 = pd.DataFrame(dv, columns=['isim', 'ID','Sequence','Drug','Time','Concentration'])
            if isim == 0:
                ipredAll = ipred_2
            else:
                ipredAll = pd.concat([ipredAll, ipred_2])
        
    elif sequence.Drug[0] == 5:
        # VCM (Meiji)
        tvCL1 = 0.0339
        tvCL2 = 0.243
        tvCL3 = 3.95
        tvV1 = 0.720
        tvQ  = 8.36
        tvV2 = 78.0
        
        wCL = 0.390
        wV1 = 0.101
        wQ  = 0.174
        wV2 = 0.819

        theta = [tvCL1, tvCL2, tvCL3, tvV1, tvQ, tvV2]
        omega = [wCL, wV1, wQ, wV2]

        for isim in range(NSIM):
            etas = np.random.randn(len(omega)) * omega
            
            _ipred = PRED_VCM_Meiji(theta, etas)
            dv["isim"]          = isim
            dv["Concentration"] = _ipred[0]
            ipred_2 = pd.DataFrame(dv, columns=['isim', 'ID','Sequence','Drug','Time','Concentration'])
            if isim == 0:
                ipredAll = ipred_2
            else:
                ipredAll = pd.concat([ipredAll, ipred_2])
        
    elif sequence.Drug[0] == 6:
        # VCM (Paediatrics)
        tvCL1 = 0.119
        tvCL2 = 0.0619
        tvCL3 = 0.00508
        tvV = 0.522
        
        wCL = 0.396
        wV  = 0.188
        
        theta = [tvCL1, tvCL2, tvCL3, tvV]
        omega = [wCL, wV]

        for isim in range(NSIM):
            etas = np.random.randn(len(omega)) * omega
            
            _ipred = PRED_VCM_Paediatrics(theta, etas)
            dv["isim"]          = isim
            dv["Concentration"] = _ipred[0]
            ipred_2 = pd.DataFrame(dv, columns=['isim', 'ID','Sequence','Drug','Time','Concentration'])
            if isim == 0:
                ipredAll = ipred_2
            else:
                ipredAll = pd.concat([ipredAll, ipred_2])
        
    elif sequence.Drug[0] == 7:
        # ABK
        tvCL1 = 0.0319
        tvCL2 = 26.5
        tvCL3 = 0.0130
        tvCL4 = 0.0342
        tvV11 = 0.272
        tvV12 = 1.19
        tvQ  = 3.84
        tvV2 = 50.6
        
        wCL = 0.388
        wV1 = 0.371
        wV2 = 1.646
        
        theta = [tvCL1, tvCL2, tvCL3, tvCL4, tvV11, tvV12, tvQ, tvV2]
        omega = [wCL, wV1, wV2]
        
        for isim in range(NSIM):
            etas = np.random.randn(len(omega)) * omega
            
            _ipred = PRED_ABK(theta, etas)
            dv["isim"]          = isim
            dv["Concentration"] = _ipred[0]
            ipred_2 = pd.DataFrame(dv, columns=['isim', 'ID','Sequence','Drug','Time','Concentration'])
            if isim == 0:
                ipredAll = ipred_2
            else:
                ipredAll = pd.concat([ipredAll, ipred_2])

    elif sequence.Drug[0] == 8:
        # DAP
        tvCL1 = 0.8016
        tvCL2 = 0.2026
        tvV = 12.29
        
        wCL = 0.2074
        
        theta = [tvCL1, tvCL2, tvV]
        omega = [wCL, wV]
        
        for isim in range(NSIM):
            etas = np.random.randn(len(omega)) * omega
            
            _ipred = PRED_DAP(theta, etas)
            dv["isim"]          = isim
            dv["Concentration"] = _ipred[0]
            ipred_2 = pd.DataFrame(dv, columns=['isim', 'ID','Sequence','Drug','Time','Concentration'])
            if isim == 0:
                ipredAll = ipred_2
            else:
                ipredAll = pd.concat([ipredAll, ipred_2])

    ipred_grouped = ipredAll.groupby('Time', as_index=False)
    pi = pd.DataFrame(ipred_2, columns=['ID','Sequence','Drug','Time'])
    pi.reset_index(drop=True, inplace=True)
    pi["Concentration_P5" ] = ipred_grouped['Concentration'].quantile(0.05)
    pi["Concentration_P50"] = ipred_grouped['Concentration'].quantile(0.50)
    pi["Concentration_P95"] = ipred_grouped['Concentration'].quantile(0.95)
    pi["PLT_P5" ] = ipred_grouped['PLT'].quantile(0.05)
    pi["PLT_P50"] = ipred_grouped['PLT'].quantile(0.50)
    pi["PLT_P95"] = ipred_grouped['PLT'].quantile(0.95)
    
    return(pi)

### TEIC (Astellas) ###

### TEIC (Tsuji) ###

### VCM (Shiogoni) ###

### VCM (Meiji) ###

### VCM (Paediatrics) ###

### ABK ###

### DAP ###

### LZD ###
# 解くべき ODE
def _LZD_(A, t, Rate, Tinf, Ke, K12, K21, V1, Ktr, Slope, Gamma, PLT0):
    # PK
    if t <= Tinf:
        dA1dt = Rate - (Ke + K12) * A[0] + K21 * A[2]
    else:
        dA1dt = -(Ke + K12) * A[0] + K21 * A[2]
    
    dA2dt = K12 * A[0] - K21 * A[2]
    
    # PLT
    Rform = Ktr * A[3]
    Kcirc = Ktr
    Edrug = Slope * A[0] / V1
    PDI   = 1 - Edrug
    PDS   = 1
    Fback = (A[1] / PLT0) ** Gamma
    
    dPLTformdt = Rform * Fback * PDI - Ktr * A[3]
    dTr1dt = Ktr * A[3] - Ktr * A[4]
    dTr2dt = Ktr * A[4] - Ktr * A[5]
    dTr3dt = Ktr * A[5] - Ktr * A[6]
    dPLTdt = Ktr * A[6] - Kcirc * A[1] * PDS
    
    return dA1dt, dPLTdt, dA2dt, dPLTformdt, dTr1dt, dTr2dt, dTr3dt

def LZD(t, par, args):
    Ke  = par[0]
    K12 = par[1]
    K21 = par[2]
    V1  = par[3]
    Ktr = par[4]
    Slope = par[5]
    Gamma = par[6]
    PLT0  = par[7]

    Rate = args[0]
    Tinf = args[1]
    A0  = args[2]
 
    y = A0 # 0: Cobs, 1: PLTobs, 2: PKperiph, 3-6: PLT

    A = spi.odeint(_LZD_, y, t, args=(Rate,Tinf, Ke,K12,K21,V1, Ktr,Slope,Gamma,PLT0))
    pred = A[-1,]
    return pred

def PRED_LZD(theta, par):
    #global dv
    pred_Cdrug = np.zeros(len(dv.idx))
    pred_PLT   = np.zeros(len(dv.idx))
    
    #theta = [tvV1, tvV2, tvCLnr, tvCLr, dCLnrdAge, tvQ, tvMTT, tvGamma, tvSlope]
    #par   = [hCL, hV1, hV2, hQ, hMTT, hGamma, hSlope]
    
    tvV1 = theta[0]
    tvV2 = theta[1]
    tvCLnr = theta[2]
    tvCLr  = theta[3]
    dCLnrdAge = theta[4]
    tvQ    = theta[5]
    
    hCL = par[0]
    hV1 = par[1]
    hV2 = par[2]
    hQ  = par[3]
    
    hMTT = par[4]
    hGamma = par[5]
    hSlope = par[6]
    
    tvMTT  = theta[6]
    tvGamma = theta[7]
    tvSlope = theta[8]
    
    MTT = tvMTT * np.exp(hMTT)#(1 + hMTT)
    Ktr = 4.0 / MTT
    Slope = tvSlope * np.exp(hSlope)#(1 + hSlope)
    Gamma = tvGamma * np.exp(hGamma)#(1 + hGamma)

    iLast = 0
    iDV   = -1
    _plt0 = sequence.PLT[0]
    A0 = [0., _plt0, 0., _plt0, _plt0, _plt0, _plt0]
    
    for i in sequence.idx:
        if isDV[i]:
            iDV += 1
            for iAMT in range(iLast, i):
                if isDose[iAMT]:
                    tDose = amt.Time[iAMT]
                    Dose = amt.Dose[iAMT]
                    Tinf = amt.Tinf[iAMT]
                    Rate = Dose / Tinf
    
                    FSIZEV  = (amt.BW[iAMT] / TBWstd) ** 1.0
                    FSIZECL = (amt.BW[iAMT] / TBWstd) ** 0.75
            
                    CCRstd = CCR_CG(bg.Age[0], TBWstd, amt.SCR[iAMT], bg.GenFM[0])
                    #CCRstd = (140.0 - bg.Age[0]) * TBWstd / amt.SCR[iAMT] / 72.0
                    #if bg.GenFM[0] == 'F':
                    #    CCRstd *= 0.85
                    RF = CCRstd / 100.0
        
                    CL = (tvCLnr * np.exp(dCLnrdAge * (bg.Age[0] - AGEstd)) + tvCLr * RF) * FSIZECL * np.exp(hCL)#(1 + hCL)
        
                    Q  = tvQ * FSIZECL * np.exp(hQ)#(1 + hQ)
        
                    V1 = tvV1 * FSIZEV * np.exp(hV1)#(1 + hV1)
                    V2 = tvV2 * FSIZEV * np.exp(hV2)#(1 + hV2)
        
                    Ke = CL / V1
                    K12 = Q / V1
                    K21 = Q / V2
                                       
                    par1 = [Ke, K12, K21, V1, Ktr, Slope, Gamma, _plt0]
    
                    if i == iAMT + 1:
                        _time = sequence.Time[i] - tDose
                    else:
                        _time = sequence.Time[iAMT + 1] - tDose
                    
                    pred1 = LZD(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                    A0 = pred1
                    
            pred_Cdrug[iDV] = A0[0] / V1
            pred_PLT[iDV]   = A0[1]
            
            if i < sequence.idx.max():
                Rate = 0
                _time = sequence.Time[i + 1] - sequence.Time[i]
                pred1 = LZD(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                A0 = pred1
                iLast = i + 1

    #dv
    params = [CL, V1, V2, Q, MTT, Gamma, Slope]
    return(pred_Cdrug, pred_PLT, params) #, predPLT
    
#end of PRED_LZD

def CCR_CG(age, bw, scr, genFM):
    CCR = (140.0 - age) * bw / scr / 72.0
    if genFM == 'F':
        CCR *= 0.85
    
    return(CCR)

#Population parameters

### sample for LZD ###
# PK
tvV1 = 22.9
tvV2 = 24.7
tvCLnr = 1.86
tvCLr  = 1.44
dCLnrdAge = -0.021
tvQ    = 10.9
sgPK = 0.318
wCL = math.sqrt(0.136)
wV1 = math.sqrt(2.02)
wV2 = math.sqrt(0.00250)
wQ  = math.sqrt(3.32)

# PLT
tvMTT = 113.0
tvGamma = -0.187
tvSlope = 0.00566
wMTT = math.sqrt(0.0571)
wGamma = math.sqrt(0.0942)
wSlope = math.sqrt(0.224)
sgPLT  = 0.234
### LZD ###

# Temporal assignment
theta = [tvV1, tvV2, tvCLnr, tvCLr, dCLnrdAge, tvQ, tvMTT, tvGamma, tvSlope]
omega = [wCL, wV1, wV2, wQ, wMTT, wGamma, wSlope]
sigma = [sgPK, sgPLT]

TBWstd = 70.
AGEstd = 69.

if __name__ == '__main__':
    NSIM = 2
    SEED = 2901
    
    np.random.seed(SEED)
    
    folder = 'data_input'
    
    # 患者背景、sequence データ読み込み
    bg = pd.read_csv(folder + '/bg.csv')
    sequence = pd.read_csv(folder + '/sequence.csv')

    # 日付時刻から Time を計算
    sequence["Time"] = get_Time(sequence)

    # IPRED
    ipred = pd.DataFrame(np.arange(0, int(sequence.Time.max() + 1), 24), columns=['idx'])
    ipred['Time'] = ipred.idx + 0.0
    ipred.Time[0] = 0.01
    ipred['Dose'] = 0.0
    ipred = ipred[ipred.Dose.notnull()]

    ipred = pd.concat([sequence, ipred]).sort_index(by=['Time','Dose'])
        
    sequence = ipred
    sequence.reset_index(drop=True, inplace=True)

    sequence["idx"] = range(len(sequence.ID))
    dv = sequence[sequence.Dose <= 0.0]
    dv.ID = sequence.ID[0]
    dv.Sequence = sequence.Sequence[0]
    dv.Drug = sequence.Drug[0]
    amt = sequence[sequence.Dose > 0.0]
    isDose = (sequence.Dose > 0.0)
    isDV = (sequence.Dose <= 0.0)
    
    pi = do_Population()

    # 結果を csv に出力
    folder = 'data_output'
    pi.to_csv(folder + '/pi.csv', index=False)
