# 解くべき ODE
def _VCM_Meiji_(A, t, Rate, Tinf, Ke, K12, K21):
    if t <= Tinf:
        dA1dt = Rate - (Ke + K12) * A[0] + K21 * A[1]
    else:
        dA1dt = -(Ke + K12) * A[0] + K21 * A[1]
    
    dA2dt = K12 * A[0] - K21 * A[1]
    
    return dA1dt, dA2dt

def VCM_Meiji(t, par, args):
    Ke  = par[0]
    K12 = par[1]
    K21 = par[2]

    Rate = args[0]
    Tinf = args[1]
    A0   = args[2]
 
    y = A0

    A = spi.odeint(_VCM_Meiji_, y, t, args=(Rate,Tinf, Ke,K12,K21))
    pred = A[-1,]
    return pred

def PRED_VCM_Meiji(theta, par):
    #global dv
    pred_Cdrug = np.zeros(len(dv.idx))
    
    #theta = [tvCL1, tvCL2, tvCL3, tvV1, tvQ, tvV2]
    #par   = [hCL, hV1, hQ, hV2]
    
    tvCL1 = theta[0]
    tvCL2 = theta[1]
    tvCL3 = theta[2]
    tvV1  = theta[3]
    tvQ   = theta[4]
    tvV2  = theta[5]
    
    hCL = par[0]
    hV1 = par[1]
    hQ  = par[2]
    hV2 = par[3]
    
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
                        CL = tvCL1 * CCR + tvCL2
                    else:
                        CL = tvCL3
                    CL *= np.exp(hCL)
        
                    V1 = tvV1 * amt.BW[iAMT] * np.exp(hV1)
                    Q  = tvQ                 * np.exp(hQ)
                    V2 = tvV2                * np.exp(hV2)

                    Ke = CL / V1
                    K12 = Q / V1
                    K21 = Q / V2
                                       
                    par1 = [Ke, K12, K21]
    
                    if i == iAMT + 1:
                        _time = sequence.Time[i] - tDose
                    else:
                        _time = sequence.Time[iAMT + 1] - tDose
                    
                    pred1 = VCM_Meiji(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                    A0 = pred1
                    
            pred_Cdrug[iDV] = A0[0] / V1
            
            if i < sequence.idx.max():
                Rate = 0
                _time = sequence.Time[i + 1] - sequence.Time[i]
                pred1 = VCM_Meiji(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                A0 = pred1
                iLast = i + 1

    #dv
    params = [CL, V1, Q, V2]
    return(pred_Cdrug, params)
    
#end of PRED_VCM_Meiji

def ss_VCM_Meiji(par):
    # par = [hCL, hV1, hQ, hV2]
    pred = PRED_VCM_Meiji(theta, par)
    sumsq = sum(((dv.Cdrug / pred[0] - 1) / sigma[0]) ** 2)
    sumsq += (par[0]/omega[0])**2 + (par[1]/omega[1])**2 + (par[2]/omega[2])**2 + (par[3]/omega[3])**2
    
    return sumsq

def Bayes_VCM_Meiji():
    # Bayes
    par0 = [0.0, 0.0, 0.0, 0.0] # 初期値
    
    ans = opt.minimize(ss_VCM_Meiji, par0, options={'maxiter': MAXITER, 'gtol': GTOL, 'disp': True})
    #print(ans)
    
    _pred = PRED_VCM_Meiji(theta, ans.x)
    params = _pred[2:]
    #CL = params[0][0]
    #V1 = params[0][1]
    #Q  = params[0][2]
    #V2 = params[0][3]
    
    predCdrug = _pred[0]
    
    return(params, predCdrug, ans.x)
    
### End of VCM (Meiji) ###
