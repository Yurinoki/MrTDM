# 解くべき ODE
def _VCM_Shionogi_(A, t, Rate, Tinf, Ke, K12, K21):
    if t <= Tinf:
        dA1dt = Rate - (Ke + K12) * A[0] + K21 * A[1]
    else:
        dA1dt = -(Ke + K12) * A[0] + K21 * A[1]
    
    dA2dt = K12 * A[0] - K21 * A[1]
    
    return dA1dt, dA2dt

def VCM_Shionogi(t, par, args):
    Ke  = par[0]
    K12 = par[1]
    K21 = par[2]

    Rate = args[0]
    Tinf = args[1]
    A0   = args[2]
 
    y = A0

    A = spi.odeint(_VCM_Shionogi_, y, t, args=(Rate,Tinf, Ke,K12,K21))
    pred = A[-1,]
    return pred

def PRED_VCM_Shionogi(theta, par):
    #global dv
    pred_Cdrug = np.zeros(len(dv.idx))
    
    #theta = [tvCL1, tvCL2, tvVss, tvK12, tvK21]
    #par   = [hCL, hVss, hK21]
    
    tvCL1 = theta[0]
    tvCL2 = theta[1]
    tvVss = theta[2]
    tvK12 = theta[3]
    tvK21 = theta[4]
    
    hCL  = par[0]
    hVss = par[1]
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
                    if CCR <= 85.0:
                        CL = tvCL1 * CCR
                    else:
                        CL = tvCL2
                    CL *= np.exp(hCL)
        
                    Vss = tvVss * np.exp(hVss)
                    K12 = tvK12
                    K21 = tvK21 * np.exp(hK21)
        
                    V1 = K21 / (K12 + K21) * Vss
                    Ke = CL / V1
                                       
                    par1 = [Ke, K12, K21]
    
                    if i == iAMT + 1:
                        _time = sequence.Time[i] - tDose
                    else:
                        _time = sequence.Time[iAMT + 1] - tDose
                    
                    pred1 = VCM_Shionogi(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                    A0 = pred1
                    
            pred_Cdrug[iDV] = A0[0] / V1
            
            if i < sequence.idx.max():
                Rate = 0
                _time = sequence.Time[i + 1] - sequence.Time[i]
                pred1 = VCM_Shionogi(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                A0 = pred1
                iLast = i + 1

    #dv
    params = [CL, Vss, K12, K21]
    return(pred_Cdrug, params)
    
#end of PRED_VCM_Shionogi

def ss_VCM_Shionogi(par):
    # par = [hCL, hVss, hK21]
    pred = PRED_VCM_Shionogi(theta, par)
    sumsq = sum(((dv.Cdrug / pred[0] - 1) / sigma[0]) ** 2)
    sumsq += (par[0]/omega[0])**2 + (par[1]/omega[1])**2 + (par[2]/omega[2])**2
    
    return sumsq

def Bayes_VCM_Shionogi():
    # Bayes
    par0 = [0.0, 0.0, 0.0] # 初期値
    
    ans = opt.minimize(ss_VCM_Shionogi, par0, options={'maxiter': MAXITER, 'gtol': GTOL, 'disp': True})
    #print(ans)
    
    _pred = PRED_VCM_Shionogi(theta, ans.x)
    params = _pred[2:]
    #CL  = params[0][0]
    #Vss = params[0][1]
    #K12 = params[0][2]
    #K21 = params[0][3]
    
    predCdrug = _pred[0]
    
    return(params, predCdrug, ans.x)
    
### End of VCM (Shionogi) ###
