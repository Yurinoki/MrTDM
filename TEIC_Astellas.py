# 解くべき ODE
def _TEIC_Astellas_(A, t, Rate, Tinf, Ke, K12, K21):
    if t <= Tinf:
        dA1dt = Rate - (Ke + K12) * A[0] + K21 * A[1]
    else:
        dA1dt = -(Ke + K12) * A[0] + K21 * A[1]
    
    dA2dt = K12 * A[0] - K21 * A[1]
    
    return dA1dt, dA2dt

def TEIC_Astellas(t, par, args):
    Ke  = par[0]
    K12 = par[1]
    K21 = par[2]

    Rate = args[0]
    Tinf = args[1]
    A0   = args[2]
 
    y = A0

    A = spi.odeint(_TEIC_Astellas_, y, t, args=(Rate,Tinf, Ke,K12,K21))
    pred = A[-1,]
    return pred

def PRED_TEIC_Astellas(theta, par):
    #global dv
    pred_Cdrug = np.zeros(len(dv.idx))
    
    #theta = [tvCL1, tvCL2, tvV1, tvK12, tvK21]
    #par   = [hCL, hV1, hK21]
    
    tvCL1 = theta[0]
    tvCL2 = theta[1]
    tvV1  = theta[2]
    tvK12 = theta[3]
    tvK21 = theta[4]
    
    hCL  = par[0]
    hV1  = par[1]
    hK21 = par[2]
    
    iLast = 0
    iDV   = -1
    A0 = [0., 0.]
    
    for i in sequence.idx:
        if isDV[i]:
            iDV += 1
            for iAMT in range(iLast, i):
                if isDose[iAMT]:
                    tDose = amt.Time[iAMT]
                    Dose = amt.Dose[iAMT]
                    Tinf = amt.Tinf[iAMT]
                    Rate = Dose / Tinf
                    
                    CCR = CCR_CG(bg.Age[0], amt.BW[iAMT], amt.SCR[iAMT], bg.GenFM[0])
                    CL = (tvCL1 * CCR + tvCL2 * amt.BW[iAMT]) * np.exp(hCL)
        
                    V1  = tvV1 * np.exp(hV1)
                    K12 = tvK12
                    K21 = tvK21 * np.exp(hK21)
        
                    Ke = CL / V1
                                       
                    par1 = [Ke, K12, K21]
    
                    if i == iAMT + 1:
                        _time = sequence.Time[i] - tDose
                    else:
                        _time = sequence.Time[iAMT + 1] - tDose
                    
                    pred1 = TEIC_Astellas(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                    A0 = pred1
                    
            pred_Cdrug[iDV] = A0[0] / V1
            
            if i < sequence.idx.max():
                Rate = 0
                _time = sequence.Time[i + 1] - sequence.Time[i]
                pred1 = TEIC_Astellas(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                A0 = pred1
                iLast = i + 1

    #dv
    params = [CL, V1, K12, K21]
    return(pred_Cdrug, params)
    
#end of PRED_TEIC_Astellas

def ss_TEIC_Astellas(par):
    # par = [hCL, hV1, hK21]
    pred = PRED_TEIC_Astellas(theta, par)
    sumsq = sum(((dv.Cdrug / pred[0] - 1) / sigma[0]) ** 2)
    sumsq += (par[0]/omega[0])**2 + (par[1]/omega[1])**2 + (par[2]/omega[2])**2
    
    return sumsq

def Bayes_TEIC_Astellas():
    # Bayes
    par0 = [0.0, 0.0, 0.0] # 初期値
    
    ans = opt.minimize(ss_TEIC_Astellas, par0, options={'maxiter': MAXITER, 'gtol': GTOL, 'disp': True})
    #print(ans)
    
    _pred = PRED_TEIC_Astellas(theta, ans.x)
    params = _pred[2:]
    #CL  = params[0][0]
    #V1  = params[0][1]
    #K12 = params[0][2]
    #K21 = params[0][3]
    
    predCdrug = _pred[0]
    
    return(params, predCdrug, ans.x)
    
### End of VCM (Shionogi) ###
