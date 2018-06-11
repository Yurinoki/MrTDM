# 解くべき ODE
def _TEIC_Tsuji_(A, t, Rate, Tinf, Ke, K12, K21):
    if t <= Tinf:
        dA1dt = Rate - (Ke + K12) * A[0] + K21 * A[1]
    else:
        dA1dt = -(Ke + K12) * A[0] + K21 * A[1]
    
    dA2dt = K12 * A[0] - K21 * A[1]
    
    return dA1dt, dA2dt

def TEIC_Tsuji(t, par, args):
    Ke  = par[0]
    K12 = par[1]
    K21 = par[2]

    Rate = args[0]
    Tinf = args[1]
    A0   = args[2]
 
    y = A0

    A = spi.odeint(_TEIC_Tsuji_, y, t, args=(Rate,Tinf, Ke,K12,K21))
    pred = A[-1,]
    return pred

def PRED_TEIC_Tsuji(theta, par):
    #global dv
    pred_Cdrug = np.zeros(len(dv.idx))
    
    #theta = [tvCL1, tvCL2, tvCL3, tvV11, tvV12, tvQ, tvV2]
    #par   = [hCL, hV1]
    
    tvCL1 = theta[0]
    tvCL2 = theta[1]
    tvCL3 = theta[2]
    tvV11 = theta[3]
    tvV12 = theta[4]
    tvQ   = theta[5]
    tvV2  = theta[6]
    
    hCL  = par[0]
    hV1  = par[1]
    
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
                    CL = tvCL1 * ((CCR/71.8)**tvCL2)
                    if bg.GenFM[0] == 'F':
                        CL *= tvCL3
                    CL *= np.exp(hCL)
        
                    V1  = tvV11 * ((amt.BW[iAMT]/53.2)**tvV12) * np.exp(hV1)
                    Q   = tvQ
                    V2  = tvV2
        
                    Ke = CL / V1
                                       
                    par1 = [Ke, K12, K21]
    
                    if i == iAMT + 1:
                        _time = sequence.Time[i] - tDose
                    else:
                        _time = sequence.Time[iAMT + 1] - tDose
                    
                    pred1 = TEIC_Tsuji(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                    A0 = pred1
                    
            pred_Cdrug[iDV] = A0[0] / V1
            
            if i < sequence.idx.max():
                Rate = 0
                _time = sequence.Time[i + 1] - sequence.Time[i]
                pred1 = TEIC_Tsuji(np.hstack((0, _time)), par1, [Rate, Tinf, A0])
                A0 = pred1
                iLast = i + 1

    #dv
    params = [CL, V1, Q, V2]
    return(pred_Cdrug, params)
    
#end of PRED_TEIC_Tsuji

def ss_TEIC_Tsuji(par):
    # par = [hCL, hV1, hK21]
    pred = PRED_TEIC_Tsuji(theta, par)
    sumsq = sum(((dv.Cdrug / pred[0] - 1) / sigma[0]) ** 2)

    w1 = omega[0]
    w2 = omega[1]
    w12 = omega[2]
    det = 1.0 / (w1**2 * w2**2 - w12**2)
    sumsq += det * ((w2**2)*(par[0]**2) - 2.0 * w12 * par[0] * par[1] + (w1**2)*(par[1]**2))
    #sumsq += (par[0]/omega[0])**2 + (par[1]/omega[1])**2
    
    return sumsq

def Bayes_TEIC_Tsuji():
    # Bayes
    par0 = [0.0, 0.0] # 初期値
    
    ans = opt.minimize(ss_TEIC_Tsuji, par0, options={'maxiter': MAXITER, 'gtol': GTOL, 'disp': True})
    #print(ans)
    
    _pred = PRED_TEIC_Tsuji(theta, ans.x)
    params = _pred[2:]
    #CL  = params[0][0]
    #V1  = params[0][1]
    #Q   = params[0][2]
    #V2  = params[0][3]
    
    predCdrug = _pred[0]
    
    return(params, predCdrug, ans.x)
    
### End of VCM (Shionogi) ###
