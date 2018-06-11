# 解くべき ODE
def _VCM_Paediatrics_(A, t, Rate, Tinf, Ke):
    if t <= Tinf:
        dA1dt = Rate - Ke * A[0]
    else:
        dA1dt = -Ke * A[0]
    
    return dA1dt

def VCM_Paediatrics(t, par, args):
    Ke  = par[0]

    Rate = args[0]
    Tinf = args[1]
    A0   = args[2]
 
    y = A0

    A = spi.odeint(_VCM_Paediatrics_, y, t, args=(Rate,Tinf, Ke,))
    pred = A[-1,]
    return pred

def PRED_VCM_Paediatrics(theta, par):
    #global dv
    pred_Cdrug = np.zeros(len(dv.idx))
    
    #theta = [tvCL1, tvCL2, tvCL3, tvV]
    #par   = [hCL, hV]
    
    tvCL1 = theta[0]
    tvCL2 = theta[1]
    tvCL3 = theta[2]
    tvV   = theta[3]
    
    hCL = par[0]
    hV  = par[1]
    
    iLast = 0
    iDV   = -1
    A0 = [0.,]
    
    for i in sequence.idx:
        if isDV[i]:
            iDV += 1
            for iAMT in range(iLast, i):
                if isDose[iAMT]:
                    tDose = amt.Time[iAMT]
                    Dose = amt.Dose[iAMT]
                    Tinf = amt.Tinf[iAMT]
                    Rate = Dose / Tinf
                    
                    if bg.Age[0] <= 1:
                        CL = tvCL1 + tvCL2 * (bg.Age[0] - 1.0)
                    else:
                        CL = tvCL1 + tvCL3 * (1.0 - bg.Age[0])
                    CL *= amt.BW[iAMT] * np.exp(hCL)
        
                    V = tvV * amt.BW{iAMT] * np.exp(hV)
        
                    Ke = CL / V
                                       
                    par1 = [Ke,]
    
                    if i == iAMT + 1:
                        _time = sequence.Time[i] - tDose
                    else:
                        _time = sequence.Time[iAMT + 1] - tDose
                    
                    pred1 = VCM_Paediatrics(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                    A0 = pred1
                    
            pred_Cdrug[iDV] = A0[0] / V
            
            if i < sequence.idx.max():
                Rate = 0
                _time = sequence.Time[i + 1] - sequence.Time[i]
                pred1 = VCM_Paediatrics(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                A0 = pred1
                iLast = i + 1

    #dv
    params = [CL, V]
    return(pred_Cdrug, params)
    
#end of PRED_VCM_Paediatrics

def ss_VCM_Paediatrics(par):
    # par = [hCL, hVss, hK21]
    pred = PRED_VCM_Paediatrics(theta, par)
    sumsq = sum(((dv.Cdrug / pred[0] - 1) / sigma[0]) ** 2)
    sumsq += (par[0]/omega[0])**2 + (par[1]/omega[1])**2
    
    return sumsq

def Bayes_VCM_Paediatrics():
    # Bayes
    par0 = [0.0, 0.0] # 初期値
    
    ans = opt.minimize(ss_VCM_Paediatrics, par0, options={'maxiter': MAXITER, 'gtol': GTOL, 'disp': True})
    #print(ans)
    
    _pred = PRED_VCM_Paediatrics(theta, ans.x)
    params = _pred[2:]
    #CL = params[0][0]
    #V  = params[0][1]
    
    predCdrug = _pred[0]
    
    return(params, predCdrug, ans.x)
    
### End of VCM (Paediatrics) ###
